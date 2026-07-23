"""
v4 adds num cycles
v5 adds Conv
v6 adds option for noise to weights for Linear
v7 adds option for cycle correction
v8 adds positional encodings to MHSA
v9 fixed quantize output errors
v10 splits forward and init
v11 adds option for separate dac for pos encs
v12 adds option for different weight bit precisions for different layers
v13 adds weight dropout
v14 adds weight sparsity / pruning
"""

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

from ..config import (
    ConformerPositionwiseFeedForwardQuantV4Config,
    QuantizedConformerMHSARelPosV1Config,
    ConformerConvolutionQuantV4Config,
    ConformerBlockQuantV1Config,
    ConformerEncoderQuantV1Config,
)

from .....common.memristor_layers import LinearQuant, ActivationQuantizer, QuantizedMultiheadAttention, Conv1dQuant
from torch.nn.quantized._reference.modules import Linear, Conv1d

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
        self.linear_ff = LinearQuant(
            in_features=cfg.input_dim,
            out_features=cfg.hidden_dim,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            bias=True,
            noise_config=cfg.weight_noise,
            weight_dropout=cfg.weight_dropout,
            pruning_config=cfg.weight_pruning,
        )
        self.activation = cfg.activation
        self.linear_out = LinearQuant(
            in_features=cfg.hidden_dim,
            out_features=cfg.input_dim,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            bias=True,
            noise_config=cfg.weight_noise,
            weight_dropout=cfg.weight_dropout,
            pruning_config=cfg.weight_pruning,
        )

        self.lin_1_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.lin_1_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.lin_2_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

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

    def flag_quant(self):
        self.linear_ff.initialized = True
        self.linear_out.initialized = True


    def prep_quant(self):

        self.linear_ff.weight_quantizer.set_scale_and_zp()
        self.lin_1_in_quant.set_scale_and_zp()
        try:
            from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
        except ModuleNotFoundError:
            from torch_memristor.memristor_modules import TiledMemristorLinear

        if self.linear_ff.pruning_config is not None:
            with torch.no_grad():
                self.linear_ff.weight.data = self.linear_ff.pruning_config.apply(
                    self.linear_ff.weight.data, training=False
                )
        mem_lin = TiledMemristorLinear(
            in_features=self.linear_ff.in_features,
            out_features=self.linear_ff.out_features,
            weight_precision=self.linear_ff.weight_bit_prec if not self.linear_ff.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(
            activation_quant=self.lin_1_in_quant,
            linear_quant=self.linear_ff,
            num_cycles_init=self.model_cfg.num_cycles,
            correction_settings=self.model_cfg.correction_settings,
        )
        self.linear_ff = mem_lin


        self.linear_out.weight_quantizer.set_scale_and_zp()
        self.lin_2_in_quant.set_scale_and_zp()
        if self.linear_out.pruning_config is not None:
            with torch.no_grad():
                self.linear_out.weight.data = self.linear_out.pruning_config.apply(
                    self.linear_out.weight.data, training=False
                )
        mem_lin = TiledMemristorLinear(
            in_features=self.linear_out.in_features,
            out_features=self.linear_out.out_features,
            weight_precision=self.linear_out.weight_bit_prec if not self.linear_out.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(
            activation_quant=self.lin_2_in_quant,
            linear_quant=self.linear_out,
            num_cycles_init=self.model_cfg.num_cycles,
            correction_settings=self.model_cfg.correction_settings,
        )
        self.linear_out = mem_lin

        self.lin_1_in_quant = nn.Identity()
        self.lin_2_in_quant = nn.Identity()

    def prep_torch_quant(self):
        self.linear_ff.weight_quantizer.set_scale_and_zp()
        self.linear_ff = Linear.from_float(
            self.linear_ff,
            weight_qparams={
                "qscheme": self.linear_ff.weight_quantizer.method,
                "dtype": self.linear_ff.weight_quant_dtype,
                "zero_point": self.linear_ff.weight_quantizer.zero_point,
                "scale": self.linear_ff.weight_quantizer.scale,
                "quant_min": self.linear_ff.weight_quantizer.quant_min,
                "quant_max": self.linear_ff.weight_quantizer.quant_max,
            },
        )
        self.linear_out.weight_quantizer.set_scale_and_zp()
        self.linear_out = Linear.from_float(
            self.linear_out,
            weight_qparams={
                "qscheme": self.linear_out.weight_quantizer.method,
                "dtype": self.linear_out.weight_quant_dtype,
                "zero_point": self.linear_out.weight_quantizer.zero_point,
                "scale": self.linear_out.weight_quantizer.scale,
                "quant_min": self.linear_out.weight_quantizer.quant_min,
                "quant_max": self.linear_out.weight_quantizer.quant_max,
            },
        )
        self.lin_1_in_quant = nn.Identity()
        self.lin_2_in_quant = nn.Identity()


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

    def flag_quant(self):
        self.mhsa.flag_quant()
        
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
        self.model_cfg = model_cfg
        self.pointwise_conv1 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=2 * model_cfg.channels,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            bias=True,
            noise_config=model_cfg.weight_noise,
            weight_dropout=model_cfg.weight_dropout,
            pruning_config=model_cfg.weight_pruning,
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
            noise_config=model_cfg.weight_noise,
            weight_dropout=model_cfg.weight_dropout,
        )
        self.dconv_1_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

        self.dconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

        self.pointwise_conv2 = LinearQuant(
            in_features=model_cfg.channels,
            out_features=model_cfg.channels,
            weight_bit_prec=model_cfg.weight_bit_prec,
            weight_quant_dtype=model_cfg.weight_quant_dtype,
            weight_quant_method=model_cfg.weight_quant_method,
            bias=True,
            noise_config=model_cfg.weight_noise,
            weight_dropout=model_cfg.weight_dropout,
            pruning_config=model_cfg.weight_pruning,
        )
        self.pconv_1_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

        self.pconv_1_out_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

        self.pconv_2_in_quant = ActivationQuantizer(
            bit_precision=model_cfg.activation_bit_prec,
            dtype=model_cfg.activation_quant_dtype,
            method=model_cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=model_cfg.moving_average,
        )

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
        self.pointwise_conv1.weight_quantizer.set_scale_and_zp()
        self.pconv_1_in_quant.set_scale_and_zp()
        try:
            from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear
            from synaptogen_ml.memristor_modules.conv import MemristorConv1d
        except ModuleNotFoundError:
            from torch_memristor.memristor_modules import TiledMemristorLinear, MemristorConv1d

        if self.pointwise_conv1.pruning_config is not None:
            with torch.no_grad():
                self.pointwise_conv1.weight.data = self.pointwise_conv1.pruning_config.apply(
                    self.pointwise_conv1.weight.data, training=False
                )
        mem_lin = TiledMemristorLinear(
            in_features=self.pointwise_conv1.in_features,
            out_features=self.pointwise_conv1.out_features,
            weight_precision=self.pointwise_conv1.weight_bit_prec
            if not self.pointwise_conv1.weight_bit_prec == 1.5
            else 2,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(
            activation_quant=self.pconv_1_in_quant,
            linear_quant=self.pointwise_conv1,
            num_cycles_init=self.model_cfg.num_cycles,
            correction_settings=self.model_cfg.correction_settings,
        )
        self.pointwise_conv1 = mem_lin

        self.depthwise_conv.weight_quantizer.set_scale_and_zp()
        self.dconv_1_in_quant.set_scale_and_zp()
        mem_conv = MemristorConv1d(
            in_channels=self.depthwise_conv.in_channels,
            out_channels=self.depthwise_conv.out_channels,
            kernel_size=self.depthwise_conv.kernel_size,
            stride=self.depthwise_conv.stride,
            groups=self.depthwise_conv.groups,
            weight_precision=self.depthwise_conv.weight_bit_prec
            if not self.depthwise_conv.weight_bit_prec == 1.5
            else 2,
            converter_hardware_settings=self.converter_hardware_settings,
            padding=(self.depthwise_conv.kernel_size - 1) // 2,
        )
        mem_conv.init_from_conv_quant(
            activation_quant=self.dconv_1_in_quant,
            conv_quant=self.depthwise_conv,
            num_cycles_init=self.model_cfg.num_cycles,
            correction_settings=self.model_cfg.correction_settings,
        )
        # self.depth_tmp = self.depthwise_conv
        self.depthwise_conv = mem_conv

        self.pointwise_conv2.weight_quantizer.set_scale_and_zp()
        self.pconv_2_in_quant.set_scale_and_zp()
        if self.pointwise_conv2.pruning_config is not None:
            with torch.no_grad():
                self.pointwise_conv2.weight.data = self.pointwise_conv2.pruning_config.apply(
                    self.pointwise_conv2.weight.data, training=False
                )
        mem_lin = TiledMemristorLinear(
            in_features=self.pointwise_conv2.in_features,
            out_features=self.pointwise_conv2.out_features,
            weight_precision=self.pointwise_conv2.weight_bit_prec
            if not self.pointwise_conv2.weight_bit_prec == 1.5
            else 2,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(
            activation_quant=self.pconv_2_in_quant,
            linear_quant=self.pointwise_conv2,
            num_cycles_init=self.model_cfg.num_cycles,
            correction_settings=self.model_cfg.correction_settings,
        )
        self.pointwise_conv2 = mem_lin

        self.pconv_1_in_quant = nn.Identity()
        self.pconv_2_in_quant = nn.Identity()
        self.dconv_1_in_quant = nn.Identity()

    def prep_torch_quant(self):
        self.pointwise_conv1.weight_quantizer.set_scale_and_zp()
        self.pointwise_conv1 = Linear.from_float(
            self.pointwise_conv1,
            weight_qparams={
                "qscheme": self.pointwise_conv1.weight_quantizer.method,
                "dtype": self.pointwise_conv1.weight_quant_dtype,
                "zero_point": self.pointwise_conv1.weight_quantizer.zero_point,
                "scale": self.pointwise_conv1.weight_quantizer.scale,
                "quant_min": self.pointwise_conv1.weight_quantizer.quant_min,
                "quant_max": self.pointwise_conv1.weight_quantizer.quant_max,
            },
        )
        self.depthwise_conv.weight_quantizer.set_scale_and_zp()
        self.depthwise_conv = Conv1d.from_float(
            self.depthwise_conv,
            weight_qparams={
                "qscheme": self.depthwise_conv.weight_quantizer.method,
                "dtype": self.depthwise_conv.weight_quant_dtype,
                "zero_point": self.depthwise_conv.weight_quantizer.zero_point,
                "scale": self.depthwise_conv.weight_quantizer.scale,
                "quant_min": self.depthwise_conv.weight_quantizer.quant_min,
                "quant_max": self.depthwise_conv.weight_quantizer.quant_max,
            },
        )
        self.pointwise_conv2.weight_quantizer.set_scale_and_zp()
        self.pointwise_conv2 = Linear.from_float(
            self.pointwise_conv2,
            weight_qparams={
                "qscheme": self.pointwise_conv2.weight_quantizer.method,
                "dtype": self.pointwise_conv2.weight_quant_dtype,
                "zero_point": self.pointwise_conv2.weight_quantizer.zero_point,
                "scale": self.pointwise_conv2.weight_quantizer.scale,
                "quant_min": self.pointwise_conv2.weight_quantizer.quant_min,
                "quant_max": self.pointwise_conv2.weight_quantizer.quant_max,
            },
        )
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