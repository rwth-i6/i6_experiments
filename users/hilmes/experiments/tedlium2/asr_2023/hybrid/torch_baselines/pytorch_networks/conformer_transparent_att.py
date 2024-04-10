import time
import torch
from torch import nn
from torch.onnx import export as onnx_export
from torchaudio.functional import mask_along_axis

import returnn.frontend as rf

from i6_models.assemblies.conformer.conformer_v1 import (
    ConformerEncoderV1Config,
    ConformerBlockV1Config,
    ConformerPositionwiseFeedForwardV1Config,
    ConformerConvolutionV1Config,
    ConformerMHSAV1Config,
    ConformerBlockV1,
)
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1,VGG4LayerActFrontendV1Config
from i6_models.parts.conformer.norm import LayerNormNC

from i6_models.parts.conformer.convolution import ConformerConvolutionV1
from typing import Callable, Union
from dataclasses import dataclass
from i6_models.config import ModelConfiguration, ModuleFactoryV1


from typing import Optional, Tuple

import torch

IntTupleIntType = Union[Tuple[int, int], int]


import torch
import torch.nn as nn
from abc import abstractmethod
from typing import Protocol, Tuple


def _lengths_to_padding_mask(lengths: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Convert lengths to a pytorch MHSA compatible key mask

    :param lengths: [B]
    :return: B x T, where 0 means within sequence and 1 means outside sequence
    """
    i_ = torch.arange(x.shape[1], device=lengths.device)  # [T]
    return i_[None, :] < lengths[:, None]  # [B, T],


class ConformerEncoderWithTranspAtt(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockV1(cfg.block_cfg) for _ in range(cfg.num_layers)])
        self.transparent_scales = nn.Parameter(torch.empty((cfg.num_layers + 1,)))

        torch.nn.init.constant_(self.transparent_scales, 1/(cfg.num_layers + 1))

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        transparent_weights = torch.softmax(self.transparent_scales, dim=0)  # TODO: Maybe Scale 0.001
        final = transparent_weights[0] * x
        for i, module in enumerate(self.module_list):
            x = module(x, sequence_mask)  # [B, T, F']
            final = final + (transparent_weights[i + 1] * x)

        return final, sequence_mask


def apply_spec_aug(
    input: torch.Tensor, num_repeat_time: int, max_dim_time: int, num_repeat_feat: int, max_dim_feat: int
):
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
            dropout=0.2
        )
        block_cfg = ConformerBlockV1Config(ff_cfg=ff_cfg, mhsa_cfg=mhsa_cfg, conv_cfg=conv_cfg)
        frontend_cfg = VGG4LayerActFrontendV1Config(
            in_features=80,
            conv1_channels=32,
            conv2_channels=64,
            conv3_channels=64,
            conv4_channels=32,
            conv_kernel_size=(3, 1),
            pool1_kernel_size=(1, 2),
            pool1_stride=pool_1_stride,
            activation=nn.ReLU(),
            conv_padding=None,
            pool1_padding=None,
            out_features=conformer_size,
            pool2_kernel_size=(1, 2),
            pool2_stride=None,
            pool2_padding=None,
        )

        frontend = ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_cfg)
        conformer_cfg = ConformerEncoderV1Config(num_layers=12, frontend=frontend, block_cfg=block_cfg)
        self.conformer = ConformerEncoderWithTranspAtt(cfg=conformer_cfg)

        upsample_kernel = kwargs.pop("upsample_kernel", 5)
        upsample_stride = kwargs.pop("upsample_stride", 2)
        upsample_padding = kwargs.pop("upsample_padding", 1)
        self.upsample_conv = torch.nn.ConvTranspose1d(
            in_channels=conformer_size,
            out_channels=conformer_size,
            kernel_size=upsample_kernel,
            stride=upsample_stride,
            padding=upsample_padding,
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
        if self.training:
            audio_features_masked_2 = apply_spec_aug(
                audio_features, self.spec_num_time, self.spec_max_time, self.spec_num_feat, self.spec_max_feat
            )
        else:
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
        model,
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
