"""
V3 adds option to quantize bias
V4 adds option to use observer only in training

streamable version which allows for streaming training
"""

import math

import numpy as np
import torch
from torch import nn
import copy
from typing import Tuple, Optional, List, Dict

from i6_models.parts.conformer.norm import LayerNormNC
from i6_models.config import ModuleFactoryV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1
from i6_models.util import compat

from i6_models.primitives.specaugment import specaugment_v1_by_length
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1

from returnn.torch.context import get_run_ctx

from .full_qat_v1_streamable_cfg import (
    QuantModelTrainConfigV4,
    ConformerPositionwiseFeedForwardQuantV4Config,
    QuantizedMultiheadAttentionV4Config,
    ConformerConvolutionQuantV4Config,
    ConformerBlockQuantV1Config,
    ConformerEncoderQuantV1Config,
    StreamableFeatureExtractorV1Config
)
from .full_qat_v1_streamable_modules import LinearQuant, QuantizedMultiheadAttentionStreamable, Conv1dQuant, ActivationQuantizer
from torch.nn.quantized._reference.modules import Linear, Conv1d

from ...streamable_module import StreamableModule
from ...encoders.components.frontend.streamable_vgg_act import StreamableVGG4LayerActFrontendV1
from ...encoders.components.feature_extractor.streamable_feature_extractor_v1 import StreamableFeatureExtractorV1
from ...common import Mode, create_chunk_mask, add_lookahead

from .._base_streamable_ctc import StreamableCTC as Model
from ...trainers import train_handler



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

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: shape [B,T,F], F=input_dim
        :return: shape [B,T,F], F=input_dim
        """
        tensor = self.layer_norm_in_quant(tensor)
        tensor = self.layer_norm(tensor)
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


###############################################################################################################
# NOTE: now streamable
class ConformerMHSAQuantStreamable(StreamableModule):
    """
    Conformer multi-headed self-attention module with streamable mhsa

    Note: Needs to be a StreamableModule so that the mode of QuantizedMultiHeadAttentionStreamable is set appropriately.
    """

    def __init__(self, cfg: QuantizedMultiheadAttentionV4Config):

        super().__init__()

        self.layernorm = torch.nn.LayerNorm(cfg.input_dim)
        self.mhsa = QuantizedMultiheadAttentionStreamable(cfg=cfg)
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

    def forward_offline(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F) or (B, N, C, F) if we come from self.forward_streaming
        :param sequence_mask: bool mask of shape (B, T) or (B, N, C), True signals within sequence, False outside, will be inverted
        which will be applied/added to dot product, used to mask padded key positions out
        """
        inv_sequence_mask = compat.logical_not(sequence_mask)
        inv_attn_mask = None if attn_mask is None else compat.logical_not(attn_mask)

        input_tensor = self.layer_norm_in_quant(input_tensor)
        output_tensor = self.layernorm(input_tensor)  # [B,T,F] or [B,N,C,F] (but we only do layernorm across last dim so its fine)
        output_tensor = self.layer_norm_out_quant(output_tensor)

        output_tensor, _ = self.mhsa(
            output_tensor, output_tensor, output_tensor, sequence_mask=inv_sequence_mask, attn_mask=inv_attn_mask
        )  # [B,T,F] or [B,N,C,F]
        output_tensor = torch.nn.functional.dropout(output_tensor, p=self.dropout, training=self.training)  # [B,T,F]

        return output_tensor
    
    def forward_streaming(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        :param input_tensor: (B, N, C, F)
        :param sequence_mask: (B, N, C)
        :return: (B, N, C, F)
        """
        return self.forward_offline(input_tensor, sequence_mask, attn_mask)
    
    def infer(
            self, x: torch.Tensor, seq_mask: torch.Tensor, ext_chunk_sz: int,
    ) -> torch.Tensor:
        """
        :param x: chunk, carryover and future frames, with shape [t, F]
        :param seq_mask:
        :param ext_chunk_sz: number of chunk frames and future frames = C+R
        :return: chunk with shape [C+R, F] where each feature attended to the carryover and future context
        """
        # x.shape: [t, F]
        attn_mask = torch.ones(x.size(0), x.size(0), device=x.device, dtype=torch.bool)
        y = self.forward_offline(
            input_tensor=x[None, None], sequence_mask=seq_mask[None, None], attn_mask=attn_mask)  # [1, 1, t, F]
        
        return y[0, 0, -ext_chunk_sz:]  # [C+R, F]

    def prep_quant(self, extra_act_quant: bool, decompose: bool):
        self.mhsa.prep_quant(extra_act_quant, decompose=decompose)

    def prep_dequant(self):
        self.mhsa.prep_dequant()


###############################################################################################################
# NOTE: now streamable
class ConformerConvolutionQuantStreamable(StreamableModule):
    """
    Conformer convolution module with support for streaming training and inference.
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

    def forward_offline(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        :param tensor: input tensor of shape [B,T,F]
        :return: torch.Tensor of shape [B,T,F]
        """
        tensor = self.layer_norm_in_quant(tensor)
        tensor = self.layer_norm(tensor)
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
        
    def forward_streaming(self, tensor:torch.Tensor, lookahead_size: int, carry_over_size: int):
        """
        Transform chunks into "carryover + chunk" and call self.forward_offline to compute causal convolution.

        :param tensor: [B, N, C+R, F] = [batch_size, number_of_chunks, chunk_size+future_context_size, conformer_dim]
        :param lookahead_size: the future acoustic context size, i.e. number of future frames per chunk = R
        :param carry_over_size: number of past chunks we may convolve over
        :return: [B, N, C+R, F]

        B: batch size
        N: number of chunks
        C+R: chunk size (+ future acoustic context size), i.e. the "extended" chunk size
        F: feature dimension
        """
        assert tensor.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = tensor.shape
        kernel_radius = self.depthwise_conv.kernel_size // 2  # = KRN//2

        # tensor to be filled and passed to forward_offline
        conv_in = torch.zeros(
            bsz, num_chunks, kernel_radius + chunk_sz, tensor.size(-1),
            device=tensor.device
        )

        # we remove future-acoustic-context (fac) as conv convolves over multiple past chunks w/o their fac
        # FIXME: why didnt i just do:
        # tensor = tensor[:, :, :-lookahead_size].contiguous()
        tensor = tensor.flatten(1, 2)  # [B, N*(C+R), F]
        chunks_no_fac = tensor.unfold(
            1, chunk_sz - lookahead_size, chunk_sz
        ).swapaxes(-2, -1)  # [B, N, C, F]


        for i in range(num_chunks):
            if i > 0:
                # calc how many past chunks needed for conv
                conv_carry = math.ceil(kernel_radius / (chunk_sz - lookahead_size))
                # don't go over predefined carryover
                conv_carry = min(carry_over_size, conv_carry)
                carry_no_fac = chunks_no_fac[:, max(0, i - conv_carry): i].flatten(1, 2)
                carry_no_fac = carry_no_fac[:, :kernel_radius]

                conv_in[:, i, -chunk_sz - carry_no_fac.size(1):-chunk_sz] = carry_no_fac

            t_step = i * chunk_sz
            # add chunk itself
            conv_in[:, i, -chunk_sz:] = tensor[:, t_step: t_step + chunk_sz]

        conv_in = conv_in.flatten(0, 1)  # [B*N, KRN//2 + C+R, F]  (KRN is the kernel_size)

        out = self.forward_offline(conv_in)
        out = out[:, -chunk_sz:]  # remove kernel_radius and get [B*N, C+R, F]
        out = out.view(bsz, num_chunks, chunk_sz, -1)  # [B, N, C+R, F]

        return out

    def infer(
            self, x: torch.Tensor, states: Optional[List[torch.Tensor]], chunk_sz: int, lookahead_sz: int
    ) -> torch.Tensor:
        """
        :param x: the current chunk
        :param states: cached previous chunks of current layer (carryover)
        :param chunk_sz:
        :param lookahead_sz:
        """
        if states is not None:
            states_no_fac = [layer_out[:-lookahead_sz] for layer_out in states]  # remove future-acoustic-context and build carryover
            x = torch.cat((*states_no_fac, x), dim=0).unsqueeze(0)  # combine carryover and chunk like in self.forward_streaming
            x = self.forward_offline(x)[:, -chunk_sz:]  # [1, C+R, F]
        else:
            x = x.unsqueeze(0)
            x = self.forward_offline(x)  # [1, C+R, F]

        return x.squeeze(0)

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


###############################################################################################################
# NOTE: now streamable
class ConformerBlockQuantStreamable(StreamableModule):
    """
    Streamable Conformer block module
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
        self.mhsa = ConformerMHSAQuantStreamable(cfg=cfg.mhsa_cfg)
        self.conv = ConformerConvolutionQuantStreamable(model_cfg=cfg.conv_cfg)
        self.ff2 = ConformerPositionwiseFeedForwardQuant(cfg=cfg.ff_cfg)
        self.final_layer_norm = torch.nn.LayerNorm(cfg.ff_cfg.input_dim)

    def forward_offline(self, x: torch.Tensor, /, sequence_mask: torch.Tensor) -> torch.Tensor:
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
        x = self.final_layer_norm(x)  # [B, T, F]
        x = self.ln_out_quant(x)

        return x
    
    def forward_streaming(
        self, x: torch.Tensor, /, sequence_mask: torch.Tensor, attn_mask: torch.Tensor, 
        lookahead_size: int, carry_over_size: int
    ):
        """
        :param x: [B, N, C+R, F']
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside (e.g. padding), shape: [B, N, C+R]
        :param attn_mask: expecting a causal mask that prevents chunks from attending to future chunks with shape [N*(C+R), N*(C+R)]
        :param lookahead_size: number of future frames per chunk (R)
        :param carry_over_size: number of past chunks we may depend on per block (i.e. attend to or convolve over)
        :return: [B, N, C+R, F']
        """
        assert x.dim() == 4, ""

        bsz, num_chunks, chunk_sz, _ = x.shape

        y = self.ff1(x)
        y = self.add_1_in_quant(y)
        x = self.add_1_in_quant(x)
        x = 0.5 * y + x  # [B, N, C+R, F']
        x = self.add_1_out_quant(x)
        y = self.mhsa(x, sequence_mask, attn_mask)
        y = self.add_2_in_quant(y)
        x = self.add_2_in_quant(x)
        x = y + x  # [B, N, C+R, F']
        x = x.masked_fill((~sequence_mask.unsqueeze(-1)), 0.0)  # convolution of previous layer might have overwritten 0-padding
        y = self.conv(x, lookahead_size, carry_over_size)
        y = self.add_3_in_quant(y)
        x = self.add_3_in_quant(x)
        x = y + x  # [B, N, C+R, F']
        x = self.add_3_out_quant(x)
        y = self.ff2(x)
        y = self.add_4_in_quant(y)
        x = self.add_4_in_quant(x)
        x = 0.5 * y + x  # [B, N, C+R, F']
        x = self.add_4_out_quant(x)
        x = self.ln_in_quant(x)
        x = self.final_layer_norm(x)  # [B, N, C+R, F']
        x = self.ln_out_quant(x)

        # FIXME: unnecessary reshape
        x = x.reshape(bsz, num_chunks, chunk_sz, x.size(-1))  # [B, N, C+R, F']

        return x

    def infer(
            self,
            input: torch.Tensor,
            sequence_mask: torch.Tensor,
            states: Optional[List[torch.Tensor]],
            curr_layer: Optional[torch.Tensor],
            lookahead_size: int,
    ) -> torch.Tensor:
        """
        Compute encoder block outputs based on previously cached chunks (states) and current chunk of subsampled features.

        :param input: chunk outputs with shape [C+R, F'], where C, R are the chunk size and the lookahead size in #frames respectively
        :param sequence_mask:
        :param states: encoder block outputs of the previous chunk at the previous layer (the carryover for mhsa)
        :param curr_layer: encoder block outputs of the previous chunk at the current layer (needed for convolution)
        :param lookahead_size: R
        :return: encoder block outputs of the current chunk [C+R, F']
        """
        ext_chunk_sz = input.size(0)

        if states is not None:
            # combine carryover and chunk: [C+R, F'] -> [(K+1)*(C+R), F'] = [t, F'] where K is the carryover size
            all_curr_chunks = torch.cat((*states, input), dim=0) 

            # build sequence_mask for mhsa
            seq_mask = torch.ones(all_curr_chunks.size(0), device=input.device, dtype=bool).view(-1, ext_chunk_sz)  # (K+1, C+R)
            if lookahead_size > 0:
                seq_mask[:-1, -lookahead_size:] = False  # we want to ignore all fac except the one of current chunk
            seq_mask[-1] = sequence_mask[0]
            seq_mask = seq_mask.flatten()  # [t]
        else:
            all_curr_chunks = input  # [C+R, F']
            seq_mask = sequence_mask.flatten()  # [C+R]

        #
        # block forward computation
        #
        x = all_curr_chunks

        # x = 0.5 * self.ff1(all_curr_chunks) + all_curr_chunks  # [t, F']
        y = self.ff1(x)
        y = self.add_1_in_quant(y)
        x = self.add_1_in_quant(x)
        x = 0.5 * y + x  # [t, F']       
        x = self.add_1_out_quant(x)

        # x = self.mhsa.infer(x, seq_mask=seq_mask, ext_chunk_sz=ext_chunk_sz) + x[-ext_chunk_sz:]  # [C+R, F']
        y = self.mhsa.infer(x, seq_mask=seq_mask, ext_chunk_sz=ext_chunk_sz)
        y = self.add_2_in_quant(y)
        x = self.add_2_in_quant(x)
        x = y + x[-ext_chunk_sz:]  # [C+R, F']

        x = x.masked_fill((~sequence_mask[0, :, None]), 0.0)
        # x = self.conv.infer(x, states=curr_layer, chunk_sz=ext_chunk_sz, lookahead_sz=lookahead_size) + x  # [C+R, F']
        y = self.conv.infer(x, states=curr_layer, chunk_sz=ext_chunk_sz, lookahead_sz=lookahead_size)
        y = self.add_3_in_quant(y)
        x = self.add_3_in_quant(x)
        x = y + x  # [C+R, F']
        x = self.add_3_out_quant(x)

        # x = 0.5 * self.ff2(x) + x  # [C+R, F']
        y = self.ff2(x)
        y = self.add_4_in_quant(y)
        x = self.add_4_in_quant(x)
        x = 0.5 * y + x  # [C+R, F']
        x = self.add_4_out_quant(x)

        # x = self.final_layer_norm(x)  # [C+R, F']
        x = self.ln_in_quant(x)
        x = self.final_layer_norm(x)  # [C+R, F']
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


###############################################################################################################
# NOTE: now streamable
class ConformerEncoderQuantStreamable(StreamableModule):
    """
    Implementation of a streamable Conformer. 
    Derived from the convolution-augmented Transformer (short Conformer), as in the original publication.
    The model consists of a frontend and a stack of N conformer blocks and allows for streaming training and decoding.
    C.f. https://arxiv.org/pdf/2005.08100.pdf
    """

    def __init__(self, cfg: ConformerEncoderQuantV1Config):
        """
        :param cfg: conformer encoder configuration with subunits for frontend and conformer blocks
        """
        super().__init__()

        self.frontend: StreamableVGG4LayerActFrontendV1 = cfg.frontend()
        self.module_list = torch.nn.ModuleList([ConformerBlockQuantStreamable(cfg.block_cfg) for _ in range(cfg.num_layers)])

    def forward_offline(self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
    
    def forward_streaming(
            self, data_tensor: torch.Tensor, sequence_mask: torch.Tensor,
            chunk_size: int, lookahead_size: int, carry_over_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param data_tensor: input tensor of shape [B, N, C', F]
        :param sequence_mask: mask tensor where 0 defines positions within the sequence and 1 outside (e.g. padding), shape: [B, N, C']
        :param chunk_size:
        :param lookahead_size: number of future frames per chunk (R)
        :param carry_over_size: number of past chunks we may depend on per block (i.e. attend to or convolve over)
        :return: (output, out_seq_mask)
            where output is torch.Tensor of shape [B, C' * N, F'],
            out_seq_mask is a torch.Tensor of shape [B, C' * N]

        
        F: input feature dim, F': internal and output feature dim
        C': data chunk size, C: down-sampled chunk size (internal chunk size)
        N: number of chunks per sequence
        R: number of future subsampled frames each chunk gets appended
        """
        # data_tensor, sequence_mask = self.feature_extraction(raw_audio, raw_audio_len, chunk_size)

        batch_sz, num_chunks, _, _ = data_tensor.shape

        x, sequence_mask = self.frontend(data_tensor, sequence_mask)  # [B, N, C', F] -> [B, N, C, F']

        # FIXME: unnecessary
        x = x.view(batch_sz, num_chunks, -1, x.size(-1))
        sequence_mask = sequence_mask.view(batch_sz, num_chunks, sequence_mask.size(-1))

        # [B, N, C, F'] -> [B, N, C+R, F']
        x, sequence_mask = add_lookahead(x, sequence_mask=sequence_mask, lookahead_size=lookahead_size)

        attn_mask = create_chunk_mask(
            seq_len=(x.size(1) * x.size(2)),
            chunk_size=x.size(2) - lookahead_size,
            lookahead_size=lookahead_size,
            carry_over_size=carry_over_size,
            device=x.device
        )

        for module in self.module_list:
            x = module(
                x, sequence_mask, attn_mask=attn_mask,
                lookahead_size=lookahead_size, carry_over_size=carry_over_size
            )  # [B, N, C+R, F']

        # remove lookahead frames from every chunk
        if lookahead_size > 0:
            x = x[:, :, :-lookahead_size].contiguous()
            sequence_mask = sequence_mask[:, :, :-lookahead_size].contiguous()

        x = x.flatten(1, 2)  # [B, C*N, F']
        sequence_mask = sequence_mask.flatten(1, 2)  # [B, C*N]

        return x, sequence_mask

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size_frames: int,
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: [1, P*C', F], where P is the number of future chunks we need for the future frames of current chunk
        :param lengths: the number of non-padding frames [1,]
        :param states: list of encoder block outputs of previous chunks (each output having shape [C, F'])
        :param chunk_size_frames: C
        :param lookahead_size: R
        :return: encoder outputs of the current chunk, number of (non-padding) encoder outputs, intermediate encoder block outputs
        """
        assert self._mode == Mode.STREAMING, "Expected encoder to be in streaming mode for streaming inference."

        # chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
        # audio_features, audio_features_len = self.feature_extraction.infer(input, lengths, chunk_size_frames)
        audio_features, audio_features_len = input, lengths

        # [1, P*C', F] -> [P, C, F']
        x, sequence_mask = self.frontend.infer(audio_features, audio_features_len, chunk_size_frames)

        # add future acoustic context to the current chunk
        if lookahead_size > 0:
            chunk = x[0]  # the current chunk whose encoder outputs we want to compute with shape [C, F']
            chunk_seq_mask = sequence_mask[0]  # [C]

            future_ac_ctx = x[1:]  # the future acoustic context [P-1, C, F']
            fut_seq_mask = sequence_mask[1:]  # [P-1, C]

            # combine all future chunks and extract the `lookahead_size` future frames
            future_ac_ctx = future_ac_ctx.view(-1, x.size(-1))  # [(P-1)*C, F'] =: [t, F']
            fut_seq_mask = fut_seq_mask.view(-1)  # [t,]
            future_ac_ctx = future_ac_ctx[:lookahead_size]  # [R, F']
            fut_seq_mask = fut_seq_mask[:lookahead_size]  # [R,]

            # combine current chunk and its future frames
            x = torch.cat((chunk, future_ac_ctx), dim=0)  # [C+R, F']
            sequence_mask = torch.cat((chunk_seq_mask, fut_seq_mask), dim=0).unsqueeze(0)  # [1, C+R]
        else:
            x = x[0]

        # save layer outs for next chunk (state for next chunk)
        layer_outs = [x]
        prev_layer = curr_layer = None

        for i, module in enumerate(self.module_list):
            if states is not None:
                # first chunk is not provided with any previous states
                prev_layer = [prev_chunk[-1][i] for prev_chunk in states]
                curr_layer = [prev_chunk[-1][i + 1] for prev_chunk in states]

            x = module.infer(x, sequence_mask, states=prev_layer, curr_layer=curr_layer, lookahead_size=lookahead_size)
            layer_outs.append(x)

        # remove fac if any
        if lookahead_size > 0:
            x = x[:-lookahead_size]  # [C, F']
            sequence_mask = sequence_mask[:, :-lookahead_size]  # [1, C]

        x = x.unsqueeze(0)  # [1, C, F']

        return x, torch.sum(sequence_mask, dim=1), layer_outs

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


###############################################################################################################
# NOTE: now streamable
class Model(StreamableModule):
    def __init__(self, model_config_dict, **kwargs):
        epoch = kwargs.pop("epoch")
        step = kwargs.pop("step")
        if len(kwargs) >= 2:
            assert False, f"You did not use all kwargs: {kwargs}"
        elif len(kwargs) == 1:
            assert "random" in list(kwargs.keys())[0], "This must only be RETURNN random arg"

        super().__init__()
        self.train_config = QuantModelTrainConfigV4.from_dict(model_config_dict)
        # fe_config = self.train_config.feature_extraction_config
        fe_config = StreamableFeatureExtractorV1Config(
            logmel_cfg=self.train_config.feature_extraction_config, 
            specaug_cfg=self.train_config.specaug_config, specaug_start_epoch=self.train_config.specauc_start_epoch
        )
        frontend_config = self.train_config.frontend_config
        
        conformer_size = self.train_config.conformer_size
        # self.feature_extraction = LogMelFeatureExtractionV1(cfg=fe_config)

        self.feature_extraction = StreamableFeatureExtractorV1(cfg=fe_config)
        conformer_config = ConformerEncoderQuantV1Config(
            num_layers=self.train_config.num_layers,
            frontend=ModuleFactoryV1(module_class=StreamableVGG4LayerActFrontendV1, cfg=frontend_config),
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
                    activation=nn.functional.silu,
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
        self.conformer = ConformerEncoderQuantStreamable(cfg=conformer_config)

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

        # streaming relevant params
        self.chunk_size = self.train_config.chunk_size
        self.lookahead_size = self.train_config.lookahead_size
        self.carry_over_size = self.train_config.carry_over_size

        self.cfg = self.train_config  # FIXME: need this for train_step, specifically CTCTrainStepMode

    def forward_offline(
        self,
        raw_audio: torch.Tensor,
        raw_audio_len: torch.Tensor,
    ):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T, #labels + blank]
        """

        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len)
        conformer_out, out_mask = self.conformer(conformer_in, mask)
        conformer_out = self.final_dropout(conformer_out)  # FIXME: ctc_refactored was better w/o final_dropout
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

    def forward_streaming(self, raw_audio: torch.Tensor, raw_audio_len: torch.Tensor):
        """
        :param raw_audio: Audio samples as [B, T, 1]
        :param raw_audio_len: length of T as [B]
        :return: logprobs [B, T', #labels + blank]
        """
        conformer_in, mask = self.feature_extraction(raw_audio, raw_audio_len, self.chunk_size)  # [B, N, C', F]
        conformer_out, out_mask = self.conformer(
            conformer_in, mask, chunk_size=self.chunk_size, lookahead_size=self.lookahead_size, carry_over_size=self.carry_over_size,
        )  # [B, C*N, F'] = [B, T', F']
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

    def infer(
            self,
            input: torch.Tensor,
            lengths: torch.Tensor,
            states: Optional[List[List[torch.Tensor]]],
            chunk_size: int,
            lookahead_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[List[torch.Tensor]]]:
        """
        :param input: audio samples as [B=1, T, 1] where T includes future frames
        :param lengths: length of T as [B=1]
        :param states:
        :param chunk_size:
        :param lookahead_size:
        """
        assert chunk_size is not None and lookahead_size is not None
        assert input.dim() == 3 and input.size(0) == 1, "Streaming inference expects input with shape [B=1, T, 1]."

        with torch.no_grad():
            chunk_size_frames = self.feature_extraction.num_samples_to_frames(num_samples=int(chunk_size))
            conformer_in, mask = self.feature_extraction.infer(input, lengths, chunk_size_frames)
            encoder_out, encoder_out_lengths, state = self.conformer.infer(
                conformer_in, mask, states, chunk_size_frames=chunk_size_frames, lookahead_size=lookahead_size
            )
            if self.train_config.quantize_output is True:
                logits = self.lin_out_in_quant(encoder_out)
                logits = self.lin_out(logits, self.lin_out_in_quant)
                encoder_out = self.lin_out_out_quant(logits)
            else:
                encoder_out = self.final_linear(encoder_out)
            
        # return encoder_out[:, :encoder_out_lengths[0]], encoder_out_lengths, [state]
        return encoder_out, encoder_out_lengths, [state]

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


class CTCTrainStepMode(train_handler.TrainStepMode):
    """
    Handles a CTC train_step for a given mode, i.e. streaming or offline.
    This class is used for training strategies (e.g. full streaming, full offline, switching, unified, ...)
    """
    def __init__(self):
        super().__init__()

    def step(self, model: StreamableModule, data: dict, mode: Mode, scale: float) -> Tuple[Dict, int]:
        raw_audio = data["raw_audio"]  # [B, T', F]
        raw_audio_len = data["raw_audio:size1"].to("cpu")  # [B]

        labels = data["labels"]  # [B, N] (sparse)
        labels_len = data["labels:size1"]  # [B, N]

        num_phonemes = torch.sum(labels_len)

        model.set_mode_cascaded(mode)
        logprobs, audio_features_len = model(
            raw_audio=raw_audio, raw_audio_len=raw_audio_len,
        )
        model.unset_mode_cascaded()

        transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, #vocab + 1]

        ctc_loss = nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )

        mode_str = mode.name.lower()[:3]
        loss_dict = {
            "ctc.%s" % mode_str: {
                "loss": ctc_loss,
                "scale": scale
            }
        }

        return loss_dict, num_phonemes
    

def train_step(*, model: Model, data, run_ctx, **kwargs):
    train_strat: train_handler.TrainingStrategy = None
    train_step_mode = CTCTrainStepMode()
    match model.cfg.train_mode:
        case train_handler.TrainMode.UNIFIED:
            train_strat = train_handler.TrainUnified(model, train_step_mode, streaming_scale=model.cfg.streaming_scale)
        case train_handler.TrainMode.SWITCHING:
            train_strat = train_handler.TrainSwitching(model, train_step_mode, run_ctx=run_ctx)
        case train_handler.TrainMode.STREAMING:
            train_strat = train_handler.TrainStreaming(model, train_step_mode)
        case train_handler.TrainMode.OFFLINE:
            train_strat = train_handler.TrainOffline(model, train_step_mode)
        case _:
            raise NotImplementedError("Training Strategy not available.")

    loss_dict, num_phonemes = train_strat.step(data)

    for loss_key in loss_dict:
        run_ctx.mark_as_loss(
            name=loss_key,
            loss=loss_dict[loss_key]["loss"],
            inv_norm_factor=num_phonemes,
            scale=loss_dict[loss_key]["scale"]
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

    if model.cfg.train_mode == train_handler.TrainMode.OFFLINE:
        model.set_mode_cascaded(Mode.OFFLINE)
    elif model.cfg.train_mode == train_handler.TrainMode.STREAMING:
        model.set_mode_cascaded(Mode.STREAMING)
    elif model.cfg.train_mode == train_handler.TrainMode.SWITCHING:
        model.set_mode_cascaded(Mode.STREAMING if run_ctx.global_step % 2 == 0 else Mode.OFFLINE)
    else:
        raise NotImplementedError

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    model.unset_mode_cascaded()

    probs = torch.exp(logprobs)
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
