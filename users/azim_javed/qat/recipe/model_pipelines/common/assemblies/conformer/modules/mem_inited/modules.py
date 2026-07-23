"""
v4 adds num cycles
v5 adds Conv
v6 adds option for noise to weights for Linear
v7 adds option for cycle correction
v8 adds positional encodings to MHSA
v9 fixed quantize output errors
v10 splits forward and init, mem inited replaces linears with already memristor layers
v11 adds option for separate dac for pos encs
v12 adds option for different weight bit precisions for different layers
v13 adds weight dropout
v14 adds weight sparsity / pruning
"""

import math

import numpy as np
import torch
from torch import nn
import copy
from typing import Tuple, Optional, List

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.util import compat

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

# from returnn.torch.context import get_run_ctx

from ...config import (
    ConformerPositionwiseFeedForwardQuantV4Config,
    QuantizedConformerMHSARelPosV1Config,
    ConformerConvolutionQuantV4Config,
    ConformerBlockQuantV1Config,
    ConformerEncoderQuantV1Config,
)

from .....memristor_layers.mem_inited import ActivationQuantizer, QuantizedMultiheadAttention

# from lovely_tensors import monkey_patch


class ConformerPositionwiseFeedForwardQuant(nn.Module):
    """
    Conformer feedforward module
    """

    def __init__(self, cfg: ConformerPositionwiseFeedForwardQuantV4Config):
        super().__init__()
        # monkey_patch()

        self.model_cfg = cfg
        self.layer_norm = nn.LayerNorm(cfg.input_dim)

        try:
            from torch_memristor.memristor_modules import TiledMemristorLinear
        except ModuleNotFoundError:
            from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear

        self.linear_ff = TiledMemristorLinear(
            in_features=cfg.input_dim,
            out_features=cfg.hidden_dim,
            weight_precision=cfg.weight_bit_prec if not cfg.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        self.activation = cfg.activation
        self.linear_out = TiledMemristorLinear(
            in_features=cfg.hidden_dim,
            out_features=cfg.input_dim,
            weight_precision=cfg.weight_bit_prec if not cfg.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        self.lin_1_in_quant = nn.Identity()

        self.lin_1_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.lin_2_in_quant = nn.Identity()

        self.lin_2_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )
        self.dropout = cfg.dropout
        self.converter_hardware_settings = cfg.converter_hardware_settings


    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm(tensor)
        tensor = self.lin_1_in_quant(tensor)
        tensor = self.linear_ff(tensor)  # [B,T,F]
        tensor = self.lin_1_out_quant(tensor)
        tensor = self.activation(tensor)  # [B,T,F]
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        tensor = self.lin_2_in_quant(tensor)
        tensor = self.linear_out(tensor)  # [B,T,F]
        tensor = self.lin_2_out_quant(tensor)
        tensor = nn.functional.dropout(tensor, p=self.dropout, training=self.training)  # [B,T,F]
        return tensor

    def prep_quant(self):
        self.linear_ff.initialized = True
        self.linear_out.initialized = True

    def prep_torch_quant(self):
        assert False, "wrong module for this"


class ConformerMHSAQuant(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: QuantizedConformerMHSARelPosV1Config):

        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = QuantizedMultiheadAttention(cfg=cfg)
        self.dropout = cfg.dropout

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside, will be inverted
        which will be applied/added to dot product, used to mask padded key positions out
        """
        # inv_sequence_mask = compat.logical_not(sequence_mask)
        # output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor = self.mhsa(input_tensor, sequence_mask=sequence_mask)  # [B,T,F]
        # output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor

    def prep_quant(self):
        self.mhsa.prep_quant()

    def prep_torch_quant(self):
        self.mhsa.prep_torch_quant()


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
        try:
            from torch_memristor.memristor_modules import TiledMemristorLinear, MemristorConv1d
        except ModuleNotFoundError:
            from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
            from synaptogen_ml.memristor_modules.conv import MemristorConv1d

        self.model_cfg = model_cfg

        self.pointwise_conv1 = TiledMemristorLinear(
            in_features=model_cfg.channels,
            out_features=2 * model_cfg.channels,
            weight_precision=model_cfg.weight_bit_prec if not model_cfg.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=model_cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        self.depthwise_conv = MemristorConv1d(
            in_channels=model_cfg.channels,
            out_channels=model_cfg.channels,
            kernel_size=model_cfg.kernel_size,
            stride=1,
            groups=model_cfg.channels,
            weight_precision=model_cfg.weight_bit_prec if not model_cfg.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=model_cfg.converter_hardware_settings,
            padding=(model_cfg.kernel_size - 1) // 2,
        )

        self.dconv_1_in_quant = nn.Identity()

        self.dconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

        self.pointwise_conv2 = TiledMemristorLinear(
            in_features=model_cfg.channels,
            out_features=model_cfg.channels,
            weight_precision=model_cfg.weight_bit_prec if not model_cfg.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=model_cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )

        self.pconv_1_in_quant = nn.Identity()
        self.pconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

        self.pconv_2_in_quant = nn.Identity()

        self.pconv_2_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )
        self.layer_norm = nn.LayerNorm(model_cfg.channels)
        self.norm = copy.deepcopy(model_cfg.norm)
        self.dropout = nn.Dropout(model_cfg.dropout)
        self.activation = model_cfg.activation
        self.converter_hardware_settings = model_cfg.converter_hardware_settings

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm(tensor)
        tensor = self.pconv_1_in_quant(tensor)
        tensor = self.pointwise_conv1(tensor)  # [B,T,2F]
        tensor = self.pconv_1_out_quant(tensor)
        tensor = nn.functional.glu(tensor, dim=-1)  # [B,T,F]

        # conv layers expect shape [B,F,T] so we have to transpose here
        tensor = tensor.transpose(1, 2)  # [B,F,T]
        tensor = self.dconv_1_in_quant(tensor)
        tensor = self.depthwise_conv(tensor)
        tensor = self.dconv_1_out_quant(tensor)

        tensor = self.norm(tensor)
        tensor = tensor.transpose(1, 2)  # transpose back to [B,T,F]

        tensor = self.activation(tensor)
        tensor = self.pconv_2_in_quant(tensor)
        tensor = self.pointwise_conv2(tensor)
        tensor = self.pconv_2_out_quant(tensor)

        return self.dropout(tensor)

    def prep_quant(self):
        self.pointwise_conv1.initialized = True
        self.pointwise_conv2.initialized = True
        self.depthwise_conv.initialized = True

    def prep_torch_quant(self):
        assert False, "Wrong module used"

class ConformerBlockQuant(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockQuantV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

        modules = []
        ff_count = 0
        for module_name in cfg.modules:
            if module_name == "ff":
                ff_cfg = cfg.ff2_cfg if (ff_count > 0 and cfg.ff2_cfg is not None) else cfg.ff_cfg
                modules.append(ConformerPositionwiseFeedForwardQuant(cfg=ff_cfg))
                ff_count += 1
            elif module_name == "mhsa":
                modules.append(ConformerMHSAQuant(cfg=cfg.mhsa_cfg))
            elif module_name == "conv":
                modules.append(ConformerConvolutionQuant(model_cfg=cfg.conv_cfg))
            else:
                raise NotImplementedError

        self.module_list = nn.ModuleList(modules)
        self.scales = cfg.scales

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        for scale, module in zip(self.scales, self.module_list):
            if isinstance(module, ConformerMHSAQuant):
                x = scale * module(x, sequence_mask) + x
            else:
                x = scale * module(x) + x
        x = self.final_layer_norm(x)  #  [B, T, F]
        return x

    def prep_quant(self):
        for module in self.module_list:
            module.prep_quant()

    def prep_torch_quant(self):
        for module in self.module_list:
            module.prep_torch_quant()


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
        if isinstance(cfg.block_cfg, list):
            assert len(cfg.block_cfg) == cfg.num_layers
            self.module_list = torch.nn.ModuleList([ConformerBlockQuant(block_cfg) for block_cfg in cfg.block_cfg])
        else:
            self.module_list = torch.nn.ModuleList([ConformerBlockQuant(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor, return_layers: Optional[List[int]] = None) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, T', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T']
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, T, F'],
            out_seq_mask is a torch.Tensor of shape [B, T]

        F: input feature dim, F': internal and output feature dim
        T': data time dim, T: down-sampled time dim (internal time dim)
        """
        if return_layers is None:
            return_layers = [len(self.module_list) - 1]

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, T, F']

        outputs = []
        assert (
            max(return_layers) < len(self.module_list) and min(return_layers) >= 0
        ), f"invalid layer index, should be between 0 and {len(self.module_list)-1}"

        for i in range(max(return_layers) + 1):
            x = self.module_list[i](x, sequence_mask)  # [B, T, F']
            if i in return_layers:
                outputs.append(x)

        return outputs, sequence_mask

    def prep_quant(self):
        for module in self.module_list:
            module.prep_quant()

    def prep_torch_quant(self):
        for module in self.module_list:
            module.prep_torch_quant()
