import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

from i6_models.parts.conformer.convolution import ConformerConvolutionV1Config
from i6_models.parts.conformer.feedforward import ConformerPositionwiseFeedForwardV1Config
from i6_models.parts.conformer.mhsa import ConformerMHSAV1Config
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config, ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config
from i6_models.config import ModuleFactoryV1

from returnn.torch.context import get_run_ctx

def apply_spec_aug(input, num_repeat_time, max_dim_time, num_repeat_feat, max_dim_feat):
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


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    r = torch.arange(tensor.shape[1], device=get_run_ctx().device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


class Model(torch.nn.Module):
    def __init__(self, **net_kwargs):
        super().__init__()
        
        self.net_kwargs = {
            "num_repeat_time": 15,
            "max_dim_time": 20,
            "num_repeat_feat": 5,
            "max_dim_feat": 10,
        }


        frontend_config = VGG4LayerActFrontendV1Config(
            in_features=50,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),
            conv_padding=None,  # =same
            pool1_stride=(3, 1),  # pool along the time axis,
            pool1_kernel_size=(3, 1),
            pool1_padding=(1, 0),
            pool2_stride=(1, 2),  # pool along the feature axis
            pool2_kernel_size=(1, 2),
            pool2_padding=None,
            out_features=256,
            activation=nn.ReLU(),
        )
        conformer_size = 256
        conformer_config = ConformerEncoderV1Config(
            num_layers=8,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardV1Config(
                    input_dim=conformer_size,
                    hidden_dim=conformer_size,
                    dropout=0.2,
                    activation=nn.functional.silu,
                ),
                mhsa_cfg=ConformerMHSAV1Config(
                    input_dim=conformer_size,
                    num_att_heads=4,
                    att_weights_dropout=0.2,
                    dropout=0.2,
                ),
                conv_cfg=ConformerConvolutionV1Config(
                    channels=conformer_size, kernel_size=9, dropout=0.2, activation=nn.functional.silu, norm=LayerNormNC(conformer_size)
                ),
            ),
        )

        target_size = 12001

        self.conformer = ConformerEncoderV1(cfg=conformer_config)
        self.upsample_conv = torch.nn.ConvTranspose1d(in_channels=conformer_size, out_channels=conformer_size,
                                                      kernel_size=3,
                                                      stride=3, padding=0)
        self.initial_linear = nn.Linear(50, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)

    def forward(
            self,
            audio_features: torch.Tensor,
            audio_features_len: torch.Tensor,
    ):
        if self.training:
            audio_features_masked_2 = apply_spec_aug(audio_features, self.net_kwargs["num_repeat_time"],
                                                     self.net_kwargs["max_dim_time"],
                                                     self.net_kwargs["num_repeat_feat"],
                                                     self.net_kwargs["max_dim_feat"])
        else:
            audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        print(conformer_in.size(1))
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out, _ = self.conformer(conformer_in, mask)
        print(conformer_out.size(1))

        upsampled = self.upsample_conv(conformer_out.transpose(1, 2)).transpose(1, 2)  # final upsampled [B, T, F]
        print(upsampled.size(1))

        # slice for correct length
        upsampled = upsampled[:, 0:audio_features.size()[1], :]
        print(upsampled.size(1))

        upsampled_dropped = nn.functional.dropout(upsampled, p=0.1, training=self.training)
        logits = self.final_linear(upsampled_dropped)  # [B, T, F]
        logits_ce_order = torch.permute(logits, dims=(0, 2, 1))  # CE expects [B, F, T]
        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, logits_ce_order


def train_step(*, model: Model, data, run_ctx, **_kwargs):
    global scripted_model
    audio_features = data["data"]
    audio_features_len = data["data:size1"]

    audio_features_len, indices = torch.sort(audio_features_len, descending=True)
    audio_features = audio_features[indices, :, :]

    phonemes = data["classes"][indices, :]
    phonemes_len = data["classes:size1"][indices]

    log_probs, logits = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    targets_packed = nn.utils.rnn.pack_padded_sequence(phonemes, phonemes_len.to("cpu"), batch_first=True,
                                                       enforce_sorted=False)
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked, reduction="sum")
    num_frames = torch.sum(audio_features_len)

    run_ctx.mark_as_loss(name="CE", loss=loss, inv_norm_factor=num_frames)


def export_trace(*, model: Model, args, f: str):
    dummy_data = torch.randn(1, 30, 50, device="cpu")
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30

    from onnxruntime.quantization.preprocess import quant_pre_process
    onnx_export(
        model.eval(),
        (dummy_data, dummy_data_len),
        f=f,
        verbose=True,
        input_names=["data", "data_len"],
        output_names=["classes"],
        opset_version=14,
        dynamic_axes={
            # dict value: manually named axes
            "data": {0: "batch", 1: "time"},
            "data_len": {0: "batch"},
            "classes": {0: "batch", 1: "time"}
        }
    )
