"""
V3 adds option to quantize bias
V4 adds option to use observer only in training
Relu instead of Silu
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import init
import copy
from typing import Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.util import compat

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .full_qat_v1_cfg import (
    QuantModelTrainConfigV4,
    ConformerPositionwiseFeedForwardQuantV4Config,
    QuantizedMultiheadAttentionV4Config,
    ConformerConvolutionQuantV4Config,
    ConformerBlockQuantV1Config,
    ConformerEncoderQuantV1Config,
)
from .full_qat_v1_modules import LinearQuant, QuantizedMultiheadAttention, Conv1dQuant, ActivationQuantizer
from torch.nn.quantized._reference.modules import Linear, Conv1d


class ConformerPositionwiseFeedForwardQuant(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardQuantV4Config):
        super().__init__()

        self.layer_norm = nn.LayerNorm(cfg.input_dim)
        self.linear_ff = LinearQuant(
            in_features=cfg.input_dim,
            out_features=cfg.hidden_dim,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            bias=True,
            quantize_bias=cfg.quantize_bias,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.activation = cfg.activation
        self.linear_out = LinearQuant(
            in_features=cfg.hidden_dim,
            out_features=cfg.input_dim,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            bias=True,
            quantize_bias=cfg.quantize_bias,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.lin_1_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.lin_1_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.lin_2_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.lin_2_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )

        self.layer_norm_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.layer_norm_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.activation_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.activation_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.dropout = cfg.dropout
        self.layer_norm_scale = torch.nn.Parameter(torch.empty(cfg.input_dim), requires_grad=True)
        self.layer_norm_bias = torch.nn.Parameter(torch.empty(cfg.input_dim), requires_grad=True)
        init.ones_(self.layer_norm_scale)
        init.zeros_(self.layer_norm_bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm_in_quant(tensor)
        tensor = tensor - torch.mean(tensor, dim=-1, keepdim=True)
        tensor = tensor / (torch.sum(torch.abs(tensor), dim=-1, keepdim=True) / tensor.size(-1) + torch.tensor(1e-5))
        tensor = tensor * self.layer_norm_scale + self.layer_norm_bias
        tensor = self.layer_norm_out_quant(tensor)
        tensor = self.lin_1_in_quant(tensor)
        tensor = self.linear_ff(tensor, self.lin_1_in_quant)  # [B,T,F]
        tensor = self.lin_1_out_quant(tensor)
        tensor = self.activation_in_quant(tensor)
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = self.activation_out_quant(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        tensor = self.lin_2_in_quant(tensor)
        tensor = self.linear_out(tensor, self.lin_2_in_quant)  # [B,T,F]
        tensor = self.lin_2_out_quant(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        return tensor

    def prep_quant(self, extra_act_quant, decompose):
        self.linear_ff.weight_quantizer.set_scale_and_zp()
        self.linear_ff = Linear.from_float(
            self.linear_ff,
            weight_qparams={
                "qscheme": self.linear_ff.weight_quant_method,
                "dtype": self.linear_ff.weight_quant_dtype,
                "zero_point": self.linear_ff.weight_quantizer.zero_point,
                "scale": self.linear_ff.weight_quantizer.scale,
                "quant_min": self.linear_ff.weight_quantizer.quant_min,
                "quant_max": self.linear_ff.weight_quantizer.quant_max,
                "is_decomposed": decompose,
            },
        )
        self.linear_out.weight_quantizer.set_scale_and_zp()
        self.linear_out = Linear.from_float(
            self.linear_out,
            weight_qparams={
                "qscheme": self.linear_out.weight_quant_method,
                "dtype": self.linear_out.weight_quant_dtype,
                "zero_point": self.linear_out.weight_quantizer.zero_point,
                "scale": self.linear_out.weight_quantizer.scale,
                "quant_min": self.linear_out.weight_quantizer.quant_min,
                "quant_max": self.linear_out.weight_quantizer.quant_max,
                "is_decomposed": decompose,
            },
        )
        if extra_act_quant is False:
            self.lin_1_in_quant = nn.Identity()
            self.lin_1_out_quant = nn.Identity()
            self.lin_2_in_quant = nn.Identity()
            self.lin_2_out_quant = nn.Identity()

    def prep_dequant(self):
        raise NotImplementedError
        tmp = nn.Linear(self.linear_ff.in_features, self.linear_ff.out_features)
        tmp.weight = self.linear_ff.weight
        tmp.bias = self.linear_ff.bias
        self.linear_ff = tmp
        del tmp
        tmp = nn.Linear(self.linear_out.in_features, self.linear_out.out_features)
        tmp.weight = self.linear_out.weight
        tmp.bias = self.linear_out.bias
        self.linear_out = tmp
        del tmp
        self.lin_1_in_quant = nn.Identity()
        self.lin_1_out_quant = nn.Identity()
        self.lin_2_in_quant = nn.Identity()
        self.lin_2_out_quant = nn.Identity()


class ConformerMHSAQuant(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: QuantizedMultiheadAttentionV4Config):

        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = QuantizedMultiheadAttention(cfg=cfg)
        self.dropout = cfg.dropout
        self.layer_norm_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.layer_norm_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
            observer_only_in_train=cfg.observer_only_in_train,
        )
        self.layer_norm_scale = torch.nn.Parameter(torch.empty(cfg.input_dim), requires_grad=True)
        self.layer_norm_bias = torch.nn.Parameter(torch.empty(cfg.input_dim), requires_grad=True)
        init.ones_(self.layer_norm_scale)
        init.zeros_(self.layer_norm_bias)

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside, will be inverted
        which will be applied/added to dot product, used to mask padded key positions out
        """
        inv_sequence_mask = compat.logical_not(sequence_mask)
        input_tensor = self.layer_norm_in_quant(input_tensor)
        input_tensor = input_tensor - torch.mean(input_tensor, dim=-1, keepdim=True)
        output_tensor = input_tensor / (torch.sum(torch.abs(input_tensor), dim=-1, keepdim=True)  / input_tensor.size(-1)  + torch.tensor(1e-5))
        output_tensor = output_tensor * self.layer_norm_scale + self.layer_norm_bias
        output_tensor = self.layer_norm_out_quant(output_tensor)

        output_tensor, _ = self.mhsa(output_tensor, output_tensor, output_tensor, mask=inv_sequence_mask)  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor

    def prep_quant(self, extra_act_quant: bool, decompose: bool):
        self.mhsa.prep_quant(extra_act_quant, decompose=decompose)

    def prep_dequant(self):
        self.mhsa.prep_dequant()


class ConformerConvolutionQuant(nn.Module):
    """
    Conformer convolution module.
    see also: https://github.com/espnet/espnet/blob/713e784c0815ebba2053131307db5f00af5159ea/espnet/nets/pytorch_backend/conformer/convolution.py#L13

    Uses explicit padding for ONNX exportability, see:
    https://github.com/pytorch/pytorch/issues/68880
    """

    def __init__(self, model_cfg: ConformerConvolutionQuantV4Config):
        """
        :param model_cfg: model configuration for this module
        """
        super().__init__()
        model_cfg.check_valid()
        self.model_cfg = model_cfg
        self.pointwise_conv1 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=2 * model_cfg.channels,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            bias=True,
            quantize_bias=model_cfg.quantize_bias,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.depthwise_conv = Conv1dQuant(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            padding=(model_cfg.kernel_size - 1) // 2,
            groups=model_cfg.channels,
            bias=True,
            stride=1,
            dilation=1,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            quantize_bias=model_cfg.quantize_bias,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.dconv_1_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.dconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pointwise_conv2 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=model_cfg.channels,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            bias=True,
            quantize_bias=model_cfg.quantize_bias,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.pconv_1_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pconv_2_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )

        self.pconv_2_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.layer_norm_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.layer_norm_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.gelu_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.gelu_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = copy.deepcopy(model_cfg.norm)
        self.norm_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.norm_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.activation_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.activation_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
            observer_only_in_train=model_cfg.observer_only_in_train,
        )
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation
        self.layer_norm_scale = torch.nn.Parameter(torch.empty(model_cfg.channels), requires_grad=True)
        self.layer_norm_bias = torch.nn.Parameter(torch.empty(model_cfg.channels), requires_grad=True)
        init.ones_(self.layer_norm_scale)
        init.zeros_(self.layer_norm_bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm_in_quant(tensor)
        tensor = tensor - torch.mean(tensor, dim=-1, keepdim=True)
        tensor = tensor / (torch.sum(torch.abs(tensor), dim=-1, keepdim=True) / tensor.size(-1) + torch.tensor(1e-5))
        tensor = tensor * self.layer_norm_scale + self.layer_norm_bias
        tensor = self.layer_norm_out_quant(tensor)
        tensor = self.pconv_1_in_quant(tensor)
        tensor = self.pointwise_conv1(tensor, self.pconv_1_in_quant)  # [B,T,2F]
        tensor = self.pconv_1_out_quant(tensor)
        tensor = self.gelu_in_quant(tensor)
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]
        tensor = self.gelu_out_quant(tensor)

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.dconv_1_in_quant(tensor)
        tensor = self.depthwise_conv(tensor, self.dconv_1_in_quant)
        tensor = self.dconv_1_out_quant(tensor)

        tensor = self.norm_in_quant(tensor)
        tensor = self.norm(tensor)
        tensor = self.norm_out_quant(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation_in_quant(tensor)
        tensor = self.activation(tensor)
        tensor = self.activation_out_quant(tensor)
        tensor = self.pconv_2_in_quant(tensor)
        tensor = self.pointwise_conv2(tensor, self.pconv_2_in_quant)
        tensor = self.pconv_2_out_quant(tensor)

        return self.dropout(tensor)

    def prep_quant(self, extra_act_quant: bool, decompose: bool):
        self.pointwise_conv1.weight_quantizer.set_scale_and_zp()
        self.pointwise_conv1 = Linear.from_float(
            self.pointwise_conv1,
            weight_qparams={
                "qscheme": self.pointwise_conv1.weight_quant_method,
                "dtype": self.pointwise_conv1.weight_quant_dtype,
                "zero_point": self.pointwise_conv1.weight_quantizer.zero_point,
                "scale": self.pointwise_conv1.weight_quantizer.scale,
                "quant_min": self.pointwise_conv1.weight_quantizer.quant_min,
                "quant_max": self.pointwise_conv1.weight_quantizer.quant_max,
                "decompose": decompose,
            },
        )
        self.depthwise_conv.weight_quantizer.set_scale_and_zp()
        self.depthwise_conv = Conv1d.from_float(
            self.depthwise_conv,
            weight_qparams={
                "qscheme": self.depthwise_conv.weight_quant_method,
                "dtype": self.depthwise_conv.weight_quant_dtype,
                "zero_point": self.depthwise_conv.weight_quantizer.zero_point,
                "scale": self.depthwise_conv.weight_quantizer.scale,
                "quant_min": self.depthwise_conv.weight_quantizer.quant_min,
                "quant_max": self.depthwise_conv.weight_quantizer.quant_max,
                "decompose": decompose,
            },
        )
        self.pointwise_conv2.weight_quantizer.set_scale_and_zp()
        self.pointwise_conv2 = Linear.from_float(
            self.pointwise_conv2,
            weight_qparams={
                "qscheme": self.pointwise_conv2.weight_quant_method,
                "dtype": self.pointwise_conv2.weight_quant_dtype,
                "zero_point": self.pointwise_conv2.weight_quantizer.zero_point,
                "scale": self.pointwise_conv2.weight_quantizer.scale,
                "quant_min": self.pointwise_conv2.weight_quantizer.quant_min,
                "quant_max": self.pointwise_conv2.weight_quantizer.quant_max,
                "decompose": decompose,
            },
        )
        if extra_act_quant is False:
            self.pconv_1_in_quant = nn.Identity()
            self.pconv_1_out_quant = nn.Identity()
            self.dconv_1_in_quant = nn.Identity()
            self.dconv_1_out_quant = nn.Identity()
            self.pconv_2_in_quant = nn.Identity()
            self.pconv_2_out_quant = nn.Identity()

    def prep_dequant(self):
        tmp = nn.Linear(self.pointwise_conv1.in_features, self.pointwise_conv1.out_features)
        tmp.weight = self.pointwise_conv1.weight
        tmp.bias = self.pointwise_conv1.bias
        self.pointwise_conv1 = tmp
        del tmp
        tmp = nn.Conv1d(
            in_channels=self.model_cfg.channels,
            out_channels=self.model_cfg.channels,
            kernel_size=self.model_cfg.kernel_size,
            padding=(self.model_cfg.kernel_size - 1) // 2,
            groups=self.model_cfg.channels,
            bias=True,
            stride=1,
            dilation=1,
        )
        tmp.weight = self.depthwise_conv.weight
        tmp.bias = self.depthwise_conv.bias
        self.depthwise_conv = tmp
        del tmp
        tmp = nn.Linear(self.pointwise_conv2.in_features, self.pointwise_conv2.out_features)
        tmp.weight = self.pointwise_conv2.weight
        tmp.bias = self.pointwise_conv2.bias
        self.pointwise_conv2 = tmp
        del tmp
        self.pconv_1_in_quant = nn.Identity()
        self.pconv_1_out_quant = nn.Identity()
        self.dconv_1_in_quant = nn.Identity()
        self.dconv_1_out_quant = nn.Identity()
        self.pconv_2_in_quant = nn.Identity()
        self.pconv_2_out_quant = nn.Identity()


class ConformerBlockQuant(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockQuantV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.add_1_in_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_1_out_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_2_in_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_2_out_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_3_in_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_3_out_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_4_in_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.add_4_out_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.ln_in_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.ln_out_quant = ActivationQuantizer(
            bit_precision=cfg.ff_cfg.activation_bit_prec,
            dtype=cfg.ff_cfg.activation_quant_dtype,
            method=cfg.ff_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.ff_cfg.moving_average,
            observer_only_in_train=cfg.ff_cfg.observer_only_in_train,
        )
        self.ff1 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAQuant(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionQuant(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)
        self.layer_norm_scale = torch.nn.Parameter(torch.empty(cfg.ff_cfg.input_dim), requires_grad=True)
        self.layer_norm_bias = torch.nn.Parameter(torch.empty(cfg.ff_cfg.input_dim), requires_grad=True)
        init.ones_(self.layer_norm_scale)
        init.zeros_(self.layer_norm_bias)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        y = self.ff1(x)
        y = self.add_1_in_quant(y)
        x = self.add_1_in_quant(x)
        x = 0.5 * y + x  # [B, T, F]
        x = self.add_1_out_quant(x)
        y = self.mhsa(x, sequence_mask)
        y = self.add_2_in_quant(y)
        x = self.add_2_in_quant(x)
        x = y + x  # [B, T, F]
        y = self.conv(x)
        y = self.add_3_in_quant(y)
        x = self.add_3_in_quant(x)
        x = y + x  # [B, T, F]
        x = self.add_3_out_quant(x)
        y = self.ff2(x)
        y = self.add_4_in_quant(y)
        x = self.add_4_in_quant(x)
        x = 0.5 * y + x  # [B, T, F]
        x = self.add_4_out_quant(x)
        x = self.ln_in_quant(x)
        x = x - torch.mean(x, dim=-1, keepdim=True)
        x = x / (torch.sum(torch.abs(x), dim=-1, keepdim=True) / x.size(-1)  + torch.tensor(1e-5))
        x = x * self.layer_norm_scale + self.layer_norm_bias
        x = self.ln_out_quant(x)

        return x

    def prep_quant(self, extra_act_quant: bool, decompose: bool):
        self.ff1.prep_quant(extra_act_quant, decompose=decompose)
        self.mhsa.prep_quant(extra_act_quant, decompose=decompose)
        self.conv.prep_quant(extra_act_quant, decompose=decompose)
        self.ff2.prep_quant(extra_act_quant, decompose=decompose)

    def prep_dequant(self):
        self.ff1.prep_dequant()
        self.mhsa.prep_dequant()
        self.conv.prep_dequant()
        self.ff2.prep_dequant()


class ConformerEncoderQuant(nn.Module):
    """
    Implementation of the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderQuantV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockQuant(cfg.block_cfg) for _ in range(cfg.num_layers)])

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
        for module in self.module_list:
            x = module(x, sequence_mask)  # [B, T, F']

        return x, sequence_mask

    def prep_quant(self, extra_act_quant: bool, decompose: bool):
        for module in self.module_list:
            module.prep_quant(extra_act_quant, decompose=decompose)

    def prep_dequant(self):
        for module in self.module_list:
            module.prep_dequant()


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
    def __init__(self, model_config_dict, **kwargs):
        epoch = kwargs.pop("epoch")
        step = kwargs.pop("step")
        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"

        super().__init__()
        self.train_config = QuantModelTrainConfigV4.from_dict(model_config_dict)
        fe_config = self.train_config.feature_extraction_config
        frontend_config = self.train_config.frontend_config
        conformer_size = self.train_config.conformer_size
        self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)
        conformer_config = ConformerEncoderQuantV1Config(
            num_layers=self.train_config.num_layers,
            frontend=ModuleFactoryV1(module_class=VGG4LayerActFrontendV1, cfg=frontend_config),
            block_cfg=ConformerBlockQuantV1Config(
                ff_cfg=ConformerPositionwiseFeedForwardQuantV4Config(
                    input_dim=conformer_size,
                    hidden_dim=self.train_config.ff_dim,
                    dropout=self.train_config.ff_dropout,
                    activation=nn.functional.relu,
                    weight_quant_dtype=self.train_config.weight_quant_dtype,
                    weight_quant_method=self.train_config.weight_quant_method,
                    activation_quant_dtype=self.train_config.activation_quant_dtype,
                    activation_quant_method=self.train_config.activation_quant_method,
                    moving_average=self.train_config.moving_average,
                    weight_bit_prec=self.train_config.weight_bit_prec,
                    activation_bit_prec=self.train_config.activation_bit_prec,
                    quantize_bias=self.train_config.quantize_bias,
                    observer_only_in_train=self.train_config.observer_only_in_train,
                ),
                mhsa_cfg=QuantizedMultiheadAttentionV4Config(
                    input_dim=conformer_size,
                    num_att_heads=self.train_config.num_heads,
                    att_weights_dropout=self.train_config.att_weights_dropout,
                    dropout=self.train_config.mhsa_dropout,
                    weight_quant_dtype=self.train_config.weight_quant_dtype,
                    weight_quant_method=self.train_config.weight_quant_method,
                    activation_quant_dtype=self.train_config.activation_quant_dtype,
                    activation_quant_method=self.train_config.activation_quant_method,
                    activation_bit_prec=self.train_config.activation_bit_prec,
                    dot_quant_dtype=self.train_config.dot_quant_dtype,
                    dot_quant_method=self.train_config.dot_quant_method,
                    Av_quant_dtype=self.train_config.Av_quant_dtype,
                    Av_quant_method=self.train_config.Av_quant_method,
                    bit_prec_W_q=self.train_config.weight_bit_prec,
                    bit_prec_W_k=self.train_config.weight_bit_prec,
                    bit_prec_W_v=self.train_config.weight_bit_prec,
                    bit_prec_dot=self.train_config.weight_bit_prec,
                    bit_prec_A_v=self.train_config.weight_bit_prec,
                    bit_prec_W_o=self.train_config.weight_bit_prec,
                    moving_average=self.train_config.moving_average,
                    quantize_bias=self.train_config.quantize_bias,
                    observer_only_in_train=self.train_config.observer_only_in_train,
                ),
                conv_cfg=ConformerConvolutionQuantV4Config(
                    channels=conformer_size,
                    kernel_size=self.train_config.conv_kernel_size,
                    dropout=self.train_config.conv_dropout,
                    activation=nn.functional.relu,
                    norm=LayerNormNC(conformer_size),
                    weight_bit_prec=self.train_config.weight_bit_prec,
                    weight_quant_dtype=self.train_config.weight_quant_dtype,
                    weight_quant_method=self.train_config.weight_quant_method,
                    activation_bit_prec=self.train_config.activation_bit_prec,
                    activation_quant_dtype=self.train_config.activation_quant_dtype,
                    activation_quant_method=self.train_config.activation_quant_method,
                    moving_average=self.train_config.moving_average,
                    quantize_bias=self.train_config.quantize_bias,
                    observer_only_in_train=self.train_config.observer_only_in_train,
                ),
            ),
        )
        self.conformer = ConformerEncoderQuant(cfg=conformer_config)

        if self.train_config.quantize_output is True:
            self.lin_out = LinearQuant(
                in_features=self.train_config.conformer_size,
                out_features=self.train_config.label_target_size + 1,
                weight_bit_prec=self.train_config.weight_bit_prec,
                weight_quant_dtype=self.train_config.weight_quant_dtype,
                weight_quant_method=self.train_config.weight_quant_method,
                bias=True,
                quantize_bias=self.train_config.quantize_bias,
                observer_only_in_train=self.train_config.observer_only_in_train,
            )
            self.lin_out_in_quant = ActivationQuantizer(
                bit_precision=self.train_config.activation_bit_prec,
                dtype=self.train_config.activation_quant_dtype,
                method=self.train_config.activation_quant_method,
                channel_axis=1,
                moving_avrg=self.train_config.moving_average,
                observer_only_in_train=self.train_config.observer_only_in_train,
            )
            self.lin_out_out_quant = ActivationQuantizer(
                bit_precision=self.train_config.activation_bit_prec,
                dtype=self.train_config.activation_quant_dtype,
                method=self.train_config.activation_quant_method,
                channel_axis=1,
                moving_avrg=self.train_config.moving_average,
                observer_only_in_train=self.train_config.observer_only_in_train,
            )
        else:
            self.final_linear = nn.Linear(conformer_size, self.train_config.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.train_config.final_dropout)
        self.specaug_start_epoch = self.train_config.specauc_start_epoch
        self.extra_act_quant = self.train_config.extra_act_quant

        self.soft_in_quant = ActivationQuantizer(
            bit_precision=self.train_config.activation_bit_prec,
            dtype=self.train_config.activation_quant_dtype,
            method=self.train_config.activation_quant_method,
            channel_axis=1,
            moving_avrg=self.train_config.moving_average,
            observer_only_in_train=self.train_config.observer_only_in_train,
        )
        self.soft_out_quant = ActivationQuantizer(
            bit_precision=self.train_config.activation_bit_prec,
            dtype=self.train_config.activation_quant_dtype,
            method=self.train_config.activation_quant_method,
            channel_axis=1,
            moving_avrg=self.train_config.moving_average,
            observer_only_in_train=self.train_config.observer_only_in_train,
        )
        # No particular weight init!

    def forward(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """

        squeezed_features = torch.squeeze(raw_audio, dim=-1)
        with torch.no_grad():
            audio_features, audio_features_len = self.feature_extraction(squeezed_features, raw_audio_len)

            run_ctx = get_run_ctx()
            if self.training and run_ctx.epoch >= self.specaug_start_epoch:
                audio_features_masked_2 = specaugment_v1_by_length(
                    audio_features,
                    time_min_num_masks=2,
                    time_max_mask_per_n_frames=self.train_config.specaug_config.repeat_per_n_frames,
                    time_mask_max_size=self.train_config.specaug_config.max_dim_time,
                    freq_min_num_masks=2,
                    freq_mask_max_size=self.train_config.specaug_config.max_dim_feat,
                    freq_max_num_masks=self.train_config.specaug_config.num_repeat_feat,
                )
            else:
                audio_features_masked_2 = audio_features

        conformer_in = audio_features_masked_2
        # create the mask for the conformer input
        mask = mask_tensor(conformer_in, audio_features_len)

        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)
        if self.train_config.quantize_output is True:
            logits = self.lin_out_in_quant(conformer_out)
            logits = self.lin_out(logits, self.lin_out_in_quant)
            logits = self.lin_out_out_quant(logits)
        else:
            logits = self.final_linear(conformer_out)

        logits = self.soft_in_quant(logits)
        log_probs = torch.log_softmax(logits, dim=2)
        log_probs = self.soft_out_quant(log_probs)

        return log_probs, torch.sum(out_mask, dim=1)

    def prep_quant(self, decompose=False):
        print("Converting Model for efficient inference")
        print(f"Activation quantization is {self.extra_act_quant}")
        if self.train_config.quantize_output is True:
            self.lin_out.weight_quantizer.set_scale_and_zp()
            self.lin_out = Linear.from_float(
                self.lin_out,
                weight_qparams={
                    "qscheme": self.lin_out.weight_quant_method,
                    "dtype": self.lin_out.weight_quant_dtype,
                    "zero_point": self.lin_out.weight_quantizer.zero_point,
                    "scale": self.lin_out.weight_quantizer.scale,
                    "quant_min": self.lin_out.weight_quantizer.quant_min,
                    "quant_max": self.lin_out.weight_quantizer.quant_max,
                    "decompose": decompose,
                },
            )
            self.final_linear = torch.nn.Sequential(self.lin_out_in_quant, self.lin_out, self.lin_out_out_quant)
        self.conformer.prep_quant(self.extra_act_quant, decompose=decompose)

    def prep_dequant(self):
        print("Removing Quantized parts from the model")
        self.conformer.prep_dequant()


def train_step(*, model: Model, data, run_ctx, **kwargs):

    raw_audio = data["raw_audio"]  # [B, T', F]
    raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

    labels = data["labels"]  # [B, N] (sparse)
    labels_len = data["labels:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.train_config.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)


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
