import torch
import torch.nn as nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis
import numpy as np
import returnn.frontend as rf

from i6_models.config import ModuleFactoryV1
from i6_models.assemblies.conformer.conformer_v1 import (
    ConformerEncoderV1,
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.primitives.specaugment import specaugment_v1_by_length


def apply_spec_aug(input: torch.Tensor, num_repeat_time: int, max_dim_time: int, num_repeat_feat: int, max_dim_feat: int):
    """
    :param Tensor input: the input audio features (B,T,F)
    :param int num_repeat_time: number of repetitions to apply time mask
    :param int max_dim_time: number of columns to be masked on time dimension will be uniformly sampled from [0, mask_param]
    :param int num_repeat_feat: number of repetitions to apply feature mask
    :param int max_dim_feat: number of columns to be masked on feature dimension will be uniformly sampled from [0, mask_param]
    """
    for _ in range(num_repeat_time):
        input = mask_along_axis(input, mask_param=max_dim_time, mask_value=0.0, axis=1)

    for _ in range(num_repeat_feat):
        input = mask_along_axis(input, mask_param=max_dim_feat, mask_value=0.0, axis=2)
    return input

# TODO: Specaugment might be a huge difference here
def _lengths_to_padding_mask(lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    i_ = torch.arange(x.shape[1], device=lengths.device)  # [T]
    return i_[None, :] < lengths[:, None]  # [B, T],


class Model(torch.nn.Module):
    """
    Do convolution first, with softmax dropout

    """

    def __init__(self, epoch, step, **kwargs):
        super().__init__()
        conformer_size = kwargs.pop("conformer_size", 384)
        target_size = 9001

        conv_kernel_size = kwargs.pop("conv_kernel_size", 31)
        att_heads = kwargs.pop("att_heads", 4)
        ff_dim = kwargs.pop("ff_dim", 2048)

        self.spec_num_time = kwargs.pop("spec_num_time", 3)
        self.spec_max_time = kwargs.pop("spec_max_time", 20)
        self.spec_num_feat = kwargs.pop("spec_num_feat", 2)
        self.spec_max_feat = kwargs.pop("spec_max_feat", 10)
        self.spec_start_epoch = kwargs.pop("spec_start_epoch", 0)
        self.old_spec = kwargs.pop("old_spec", False)
        if self.old_spec:
            print("We will use the old specaugment")

        pool_1_stride = kwargs.pop("pool_1_stride", (2, 1))
        pool_1_kernel_size = kwargs.pop("pool_1_kernel_size", (1, 2))
        pool_1_padding = kwargs.pop("pool_1_padding", None)
        pool_2_stride = kwargs.pop("pool_2_stride", None)
        pool_2_kernel_size = kwargs.pop("pool_2_kernel_size", (1, 2))
        pool_2_padding = kwargs.pop("pool_2_padding", None)

        conv_cfg = ConformerConvolutionV1Config(
            channels=conformer_size,
            kernel_size=conv_kernel_size,
            dropout=0.2,
            activation=nn.SiLU(),
            norm=LayerNormNC(conformer_size),
        )
        mhsa_cfg = ConformerMHSAV1Config(
            input_dim=conformer_size, num_att_heads=att_heads, att_weights_dropout=0.2, dropout=0.2
        )
        ff_cfg = ConformerPositionwiseFeedForwardV1Config(
            input_dim=conformer_size,
            hidden_dim=ff_dim,
            activation=nn.SiLU(),
            dropout=0.2,
        )
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        frontend_cfg = VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 1),
            pool1_kernel_size=pool_1_kernel_size,
            pool1_stride=pool_1_stride,
            activation=nn.ReLU(),
            conv_padding=None,
            pool1_padding=pool_1_padding,
            out_features=conformer_size,
            pool2_kernel_size=pool_2_kernel_size,
            pool2_stride=pool_2_stride,
            pool2_padding=pool_2_padding,
        )

        frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)
        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)

        upsample_kernel = kwargs.pop("upsample_kernel", 5)
        upsample_stride = kwargs.pop("upsample_stride", 2)
        upsample_padding = kwargs.pop("upsample_padding", 1)
        upsample_out_padding = kwargs.pop("upsample_out_padding", 0)
        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=conformer_size,
            out_channels=conformer_size,
            kernel_size=upsample_kernel,
            stride=upsample_stride,
            padding=upsample_padding,
            output_padding=upsample_out_padding
        )
        # self.initial_linear = nn.Linear(80, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)
        self.export_mode = False
        self.prior_comp = False
        assert len(kwargs) in [0, 1]  # for some reason there is some random arg always here

    def forward(
        self,
        audio_features: torch.Tensor,
        audio_features_len: torch.Tensor,
    ):
        run_ctx = rf.get_run_ctx()
        if self.training:
            if self.old_spec:
                audio_features_masked_2 = apply_spec_aug(
                    audio_features,
                    self.spec_num_time,
                    self.spec_max_time,
                    self.spec_num_feat,
                    self.spec_max_feat
                )
            else:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.spec_num_time,
                    time_mask_max_size=self.spec_max_time,
                    freq_min_num_masks=2,
                    freq_max_num_masks=self.spec_num_feat,
                    freq_mask_max_size=self.spec_max_feat,
                )
        else:
            #if run_ctx.epoch >= self.spec_start_epoch:
            #    print("Skipped SpecAug")
            audio_features_masked_2 = audio_features

        # conformer_in = self.initial_linear(audio_features_masked_2)

        mask = _lengths_to_padding_mask(audio_features_len, audio_features)
        #mask = torch.logical_xor(mask, torch.ones_like(mask))

        conformer_out, _ = self.conformer(audio_features_masked_2, mask)

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]

        # slice for correct length
        upsampled = upsampled[:, 0 : audio_features.size()[1], :]

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.2, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


# scripted_model = None


def train_step(*, model: Model, extern_data, **_kwargs):
    global scripted_model
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]
    # from returnn.frontend import Tensor
    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]
    # if scripted_model is None:
    #     scripted_model = torch.jit.script(model)

    # distributed_model = DataParallel(model)
    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    dummy_data = torch.randn(1, 30, 80, device="cpu")
    # dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30
    #scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
    onnx_export(
        model.eval(),
        (dummy_data, dummy_data_len),
        f=model_filename,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"},
        },
    )


def prior_init_hook(run_ctx, **kwargs):
    # we are storing durations, but call it output.hdf to match
    # the default output of the ReturnnForwardJob
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx, **kwargs):
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"]  # [B]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
