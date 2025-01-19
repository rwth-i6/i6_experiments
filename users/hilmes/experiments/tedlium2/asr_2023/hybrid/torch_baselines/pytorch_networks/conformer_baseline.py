import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis
from i6_models.primitives.specaugment import specaugment_v1_by_length

import returnn.frontend as rf

from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1
from i6_models.assemblies.conformer.conformer_v1 import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
)
from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1, VGG4LayerActFrontendV1Config

from i6_models.parts.conformer.convolution import ConformerConvolutionV1
from typing import Callable, Union
from dataclasses import dataclass
from i6_models.config import ModelConfiguration, ModuleFactoryV1
from typing import Optional, Tuple


def mask_tensor(tensor: torch.Tensor, seq_len: torch.Tensor) -> torch.Tensor:
    """
    mask a tensor with a "positive" mask (boolean true means position is used)

    This function is traceable.

    :param tensor: [B,T,....]
    :param seq_len: [B]
    :return: [B,T]
    """
    seq_len = seq_len.to(device=tensor.device)
    r = torch.arange(tensor.shape[1], device=tensor.device)  # [T]
    seq_mask = torch.less(r[None, :], seq_len[:, None])  # broadcast to [B,T]
    return seq_mask


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
            input_dim=conformer_size, hidden_dim=ff_dim, activation=nn.SiLU(), dropout=0.2,
        )
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        frontend_cfg = VGG4LayerActFrontendV1Config(
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 3),  # TODO this was 3
            pool1_kernel_size=pool_1_kernel_size,
            pool1_stride=pool_1_stride,
            activation=nn.ReLU(),
            conv_padding=None,
            pool1_padding=pool_1_padding,
            pool2_kernel_size=pool_2_kernel_size,
            pool2_stride=pool_2_stride,
            pool2_padding=pool_2_padding,
            in_features=80,
            out_features=conformer_size,
        )

        frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)
        self.conformer = ConformerEncoderV1(cfg=conformer_cfg)

        upsample_kernel = kwargs.pop("upsample_kernel", 5)
        upsample_stride = kwargs.pop("upsample_stride", 2)
        upsample_padding = kwargs.pop("upsample_padding", 1)
        upsample_out_padding = kwargs.pop("upsample_out_padding", 0)
        self.no_spec = kwargs.pop("no_spec", False)
        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=conformer_size,
            out_channels=conformer_size,
            kernel_size=upsample_kernel,
            stride=upsample_stride,
            padding=upsample_padding,
            output_padding=upsample_out_padding,
        )
        # self.initial_linear = nn.Linear(80, conformer_size)
        self.final_linear = nn.Linear(conformer_size, target_size)
        self.export_mode = False
        self.prior_comp = False
        assert len(kwargs) in [0, 1]  # for some reason there is some random arg always here

    def forward(
        self, audio_features: torch.Tensor, audio_features_len: torch.Tensor,
    ):
        with torch.no_grad():
            if self.training and not self.no_spec is True:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.spec_num_time,
                    time_mask_max_size=self.spec_max_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.spec_max_feat,
                    freq_max_num_masks=self.spec_num_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        # conformer_in = self.initial_linear(audio_features_masked_2)

        mask = mask_tensor(audio_features_masked_2, audio_features_len)

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

    phonemes = extern_data["classes"].raw_tensor[indices, :].long()
    phonemes_len = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor[indices]

    log_probs, logits = model(audio_features=audio_features, audio_features_len=audio_features_len.to("cuda"),)

    targets_packed = nn.utils.rnn.pack_padded_sequence(
        phonemes, phonemes_len.to("cpu"), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    loss = nn.functional.cross_entropy(logits, targets_masked)

    rf.get_run_ctx().mark_as_loss(name="CE", loss=loss)


def export(*, model: Model, model_filename: str):
    model.export_mode = True
    torch.onnx.is_in_onnx_export()
    dummy_data = torch.randn(1, 30, 80, device="cpu")
    # dummy_data_len, _ = torch.sort(torch.randint(low=10, high=30, size=(1,), device="cpu", dtype=torch.int32), descending=True)
    dummy_data_len = torch.ones((1,), dtype=torch.int32) * 30
    # scripted_model = torch.jit.trace(model.eval(), example_inputs=(dummy_data, dummy_data_len))
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
