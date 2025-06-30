"""
"""

import math

import numpy as np
import torch
from torch import nn
import copy
from typing import Tuple

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.util import compat

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .memristor_v1_cfg import (
    QuantModelTrainConfigV4,
    ConformerPositionwiseFeedForwardQuantV4Config,
    QuantizedMultiheadAttentionV4Config,
    ConformerConvolutionQuantV4Config,
    ConformerBlockQuantV1Config,
    ConformerEncoderQuantV1Config,
)
from .memristor_v3_modules import LinearQuant, ActivationQuantizer, QuantizedMultiheadAttention, Conv1dQuant
from torch.nn.quantized._reference.modules import Conv1d


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
        )
        self.activation = cfg.activation
        self.linear_out = LinearQuant(
            in_features=cfg.hidden_dim,
            out_features=cfg.input_dim,
            weight_bit_prec=cfg.weight_bit_prec,
            weight_quant_dtype=cfg.weight_quant_dtype,
            weight_quant_method=cfg.weight_quant_method,
            bias=True,
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

    def prep_quant(self):

        self.linear_ff.weight_quantizer.set_scale_and_zp()
        self.lin_1_in_quant.set_scale_and_zp()
        from torch_memristor.memristor_modules import TiledMemristorLinear

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
        )
        self.linear_ff = mem_lin

        self.linear_out.weight_quantizer.set_scale_and_zp()
        self.lin_2_in_quant.set_scale_and_zp()
        mem_lin = TiledMemristorLinear(
            in_features=self.linear_out.in_features,
            out_features=self.linear_out.out_features,
            weight_precision=self.linear_out.weight_bit_prec if not self.linear_out.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=self.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        mem_lin.init_from_linear_quant(activation_quant=self.lin_2_in_quant, linear_quant=self.linear_out)
        self.linear_out = mem_lin
        self.lin_1_in_quant = nn.Identity()
        self.lin_2_in_quant = nn.Identity()


class ConformerMHSAQuant(torch.nn.Module):
    """
    Conformer multi-headed self-attention module
    """

    def __init__(self, cfg: QuantizedMultiheadAttentionV4Config):

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
        inv_sequence_mask = compat.logical_not(sequence_mask)
        output_tensor = self.layernorm(input_tensor)  # [B,T,F]

        output_tensor, _ = self.mhsa(output_tensor, output_tensor, output_tensor, mask=inv_sequence_mask)  # [B,T,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor

    def prep_quant(self):
        self.mhsa.prep_quant()


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

    def prep_quant(self, decompose: bool):
        self.pointwise_conv1.weight_quantizer.set_scale_and_zp()
        self.pconv_1_in_quant.set_scale_and_zp()
        from torch_memristor.memristor_modules import TiledMemristorLinear

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
        )
        self.pointwise_conv1 = mem_lin

        if not "symmetric" in self.depthwise_conv.weight_quant_method:
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
        self.pconv_2_in_quant.set_scale_and_zp()
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
        )
        self.pointwise_conv2 = mem_lin
        self.pconv_1_in_quant = nn.Identity()
        self.pconv_2_in_quant = nn.Identity()


class ConformerBlockQuant(nn.Module):
    """
    Conformer block module
    """

    def __init__(self, cfg: ConformerBlockQuantV1Config):
        """
        :param cfg: conformer block configuration with subunits for the different conformer parts
        """
        super().__init__()
        self.ff1 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.mhsa = ConformerMHSAQuant(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionQuant(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        :param x: input tensor of shape [B, T, F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside, shape: [B, T]
        :return: torch.Tensor of shape [B, T, F]
        """
        x = 0.5 * self.ff1(x) + x  # [B, T, F]
        x = self.mhsa(x, sequence_mask) + x  # [B, T, F]
        x = self.conv(x) + x  # [B, T, F]
        x = 0.5 * self.ff2(x) + x  # [B, T, F]
        x = self.final_layer_norm(x)  # [B, T, F]
        return x

    def prep_quant(self, decompose):
        self.ff1.prep_quant()
        self.mhsa.prep_quant()
        self.conv.prep_quant(decompose)
        self.ff2.prep_quant()


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

    def prep_quant(self, decompose: bool):
        for module in self.module_list:
            module.prep_quant(decompose=decompose)

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
                    activation=nn.functional.silu,
                    weight_quant_dtype=self.train_config.weight_quant_dtype,
                    weight_quant_method=self.train_config.weight_quant_method,
                    activation_quant_dtype=self.train_config.activation_quant_dtype,
                    activation_quant_method=self.train_config.activation_quant_method,
                    moving_average=self.train_config.moving_average,
                    weight_bit_prec=self.train_config.weight_bit_prec,
                    activation_bit_prec=self.train_config.activation_bit_prec,
                    converter_hardware_settings=self.train_config.converter_hardware_settings,
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
                    quant_in_linear=self.train_config.quant_in_linear,
                    converter_hardware_settings=self.train_config.converter_hardware_settings,
                ),
                conv_cfg=ConformerConvolutionQuantV4Config(
                    channels=conformer_size,
                    kernel_size=self.train_config.conv_kernel_size,
                    dropout=self.train_config.conv_dropout,
                    activation=nn.functional.silu,
                    norm=LayerNormNC(conformer_size),
                    weight_bit_prec=self.train_config.weight_bit_prec,
                    weight_quant_dtype=self.train_config.weight_quant_dtype,
                    weight_quant_method=self.train_config.weight_quant_method,
                    activation_bit_prec=self.train_config.activation_bit_prec,
                    activation_quant_dtype=self.train_config.activation_quant_dtype,
                    activation_quant_method=self.train_config.activation_quant_method,
                    moving_average=self.train_config.moving_average,
                    converter_hardware_settings=self.train_config.converter_hardware_settings,
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
            )
            self.lin_out_in_quant = ActivationQuantizer(
                bit_precision=self.train_config.activation_bit_prec,
                dtype=self.train_config.activation_quant_dtype,
                method=self.train_config.activation_quant_method,
                channel_axis=1,
                moving_avrg=self.train_config.moving_average,
            )
            self.lin_out_out_quant = ActivationQuantizer(
                bit_precision=self.train_config.activation_bit_prec,
                dtype=self.train_config.activation_quant_dtype,
                method=self.train_config.activation_quant_method,
                channel_axis=1,
                moving_avrg=self.train_config.moving_average,
            )
            self.final_linear = torch.nn.Sequential(self.lin_out_in_quant, self.lin_out, self.lin_out_out_quant)
        else:
            self.final_linear = nn.Linear(conformer_size, self.train_config.label_target_size + 1)  # + CTC blank
        self.final_dropout = nn.Dropout(p=self.train_config.final_dropout)
        self.specaug_start_epoch = self.train_config.specauc_start_epoch
        self.converter_hardware_settings = self.train_config.converter_hardware_settings
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
        logits = self.final_linear(conformer_out)

        log_probs = torch.log_softmax(logits, dim=2)

        return log_probs, torch.sum(out_mask, dim=1)

    def prep_quant(self, decompose=False):
        print("Converting Model for efficient inference")
        if self.train_config.quantize_output is True:
            self.lin_out.weight_quantizer.set_scale_and_zp()
            self.lin_out_in_quant.set_scale_and_zp()
            from torch_memristor.memristor_modules import TiledMemristorLinear

            mem_lin = TiledMemristorLinear(
                in_features=self.lin_out.in_features,
                out_features=self.lin_out.out_features * 2,
                weight_precision=self.lin_out.weight_bit_prec if not self.lin_out.weight_bit_prec == 1.5 else 2,
                converter_hardware_settings=self.converter_hardware_settings,
                memristor_inputs=128,
                memristor_outputs=128,
            )
            mem_lin.init_from_linear_quant(
                activation_quant=self.lin_out_in_quant,
                linear_quant=self.lin_out,
            )
            self.final_linear = mem_lin
        self.conformer.prep_quant(decompose=decompose)


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
