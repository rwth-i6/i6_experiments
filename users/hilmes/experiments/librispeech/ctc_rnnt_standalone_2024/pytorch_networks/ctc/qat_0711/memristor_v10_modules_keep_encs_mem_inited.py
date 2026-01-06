"""
Fixes mhsa norm
"""

import torch
from torch import nn
from torch.nn import init
import torch.ao.quantization as torch_quant
import torch.nn.functional as F
from typing import Optional, Union, Dict, Callable
from .memristor_v8_cfg import QuantizedConformerMHSARelPosV1Config
import math
from torch.ao.quantization.utils import check_min_max_valid
from returnn.torch.context import get_run_ctx
import numpy
from i6_models.util import compat
from i6_models.parts.dropout import BroadcastDropout


def get_quantization_range_from_bit_precision(bits, dtype):

    if bits == 1.5:
        quant_min = -1
        quant_max = 1
    elif bits == 2.5:
        quant_min = -3
        quant_max = 3
    elif bits == 3.5:
        quant_min = -7
        quant_max = 7
    elif dtype == torch.qint8:
        quant_min = -(2 ** (bits - 1)) + 1
        quant_max = (2 ** (bits - 1)) - 1
    elif dtype == torch.quint8:
        quant_min = 0
        quant_max = (2**bits) - 1
    else:
        raise ValueError(f"Unrecognized dtype {dtype}")

    return torch.tensor(quant_min, dtype=torch.int32), torch.tensor(quant_max, dtype=torch.int32)


class WeightQuantizer(nn.Module):
    def __init__(
        self,
        bit_precision: int,
        dtype: torch.dtype,
        method: str,
        reduce_range: bool = False,
    ):
        super().__init__()

        self.quant_min, self.quant_max = get_quantization_range_from_bit_precision(bit_precision, dtype)
        self.dtype = dtype
        self.reduce_range = reduce_range
        self.quant_fn, self.observer = None, None
        self.quant_fn, self.observer = self.__get_quant_fn_and_observer_for_method(method)
        self.scale = None
        self.zero_point = None

    def __get_quant_fn_and_observer_for_method(self, method):
        if self.quant_fn is not None and self.observer is not None:
            return self.quant_fn, self.observer
        if method == "per_tensor":
            quant_fn = torch.fake_quantize_per_tensor_affine
            observer = torch_quant.observer.MinMaxObserver(
                quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype, reduce_range=self.reduce_range
            )
        elif method == "per_tensor_symmetric":
            quant_fn = torch.fake_quantize_per_tensor_affine
            observer = torch_quant.observer.MinMaxObserver(
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.dtype,
                reduce_range=self.reduce_range,
                qscheme=torch.per_tensor_symmetric,
            )
        else:
            raise ValueError(f"Unknown quantization method: {method}!")

        return quant_fn, observer

    def forward(self, tensor: torch.Tensor):
        if self.training:
            tensor = self.observer(tensor)
        self.set_scale_and_zp()
        assert self.scale is not None and self.zero_point is not None
        tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        assert check_min_max_valid(self.observer.min_val, self.observer.max_val), "Need to init observer first"
        scale, zero_point = self.observer.calculate_qparams()
        self.scale = scale.to(dtype=torch.float32)
        self.zero_point = zero_point.to(dtype=torch.int32)


class ActivationQuantizer(nn.Module):
    def __init__(
        self,
        bit_precision: int,
        dtype: torch.dtype,
        method: str,
        channel_axis: Optional[int],
        moving_avrg: Optional[float],  # default if enabled should be 0.01, if set enables moving average
        reduce_range: bool = False,
    ):
        super().__init__()
        self.quant_min, self.quant_max = get_quantization_range_from_bit_precision(bit_precision, dtype)
        self.dtype = dtype
        self.channel_axis = channel_axis
        self.moving_avrg = moving_avrg
        self.reduce_range = reduce_range
        self.quant_fn, self.observer, self.base_observer_args = None, None, None
        self.quant_fn, self.observer, self.base_observer_args = self.__get_quant_fn_and_observer_for_method(method)
        self.zero_point = None
        self.scale = None

    def __get_quant_fn_and_observer_for_method(self, method):
        if all(x is not None for x in [self.quant_fn, self.base_observer_args, self.observer]):
            return self.quant_fn, self.base_observer_args, self.observer
        if method == "per_tensor":
            quant_fn = torch.fake_quantize_per_tensor_affine
            base_observer_args = [self.quant_min, self.quant_max]
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAverageMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                )
            else:
                observer = torch_quant.observer.MinMaxObserver(
                    quant_min=self.quant_min, quant_max=self.quant_max, dtype=self.dtype, reduce_range=self.reduce_range
                )
        elif method == "per_channel":
            quant_fn = torch.fake_quantize_per_channel_affine
            base_observer_args = [self.channel_axis, self.quant_min, self.quant_max]
            assert self.channel_axis is not None
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAveragePerChannelMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    ch_axis=self.channel_axis,
                    reduce_range=self.reduce_range,
                )
            else:
                observer = torch_quant.observer.PerChannelMinMaxObserver(
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    ch_axis=self.channel_axis,
                )
        elif method == "per_tensor_symmetric":
            quant_fn = torch.fake_quantize_per_tensor_affine
            base_observer_args = [self.quant_min, self.quant_max]
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAverageMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    qscheme=torch.per_tensor_symmetric,
                )
            else:
                observer = torch_quant.observer.MinMaxObserver(
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    qscheme=torch.per_tensor_symmetric,
                )
        else:
            raise ValueError(f"Unknown quantization method: {method}!")

        return quant_fn, observer, base_observer_args

    def forward(self, tensor: torch.Tensor):
        if self.training:
            tensor = self.observer(tensor)
        self.set_scale_and_zp()
        assert (
            self.scale is not None and self.zero_point is not None
        ), "Need to calibrate before applying quant, disable apply_calibration"
        with torch.autocast(device_type=tensor.device.type, enabled=False, dtype=torch.bfloat16):
            tensor = self.quant_fn(tensor.float(), self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        assert check_min_max_valid(self.observer.min_val, self.observer.max_val), "Need to init observer first"
        scale, zero_point = self.observer.calculate_qparams()
        self.scale = scale.to(dtype=torch.float32)
        self.zero_point = zero_point.to(dtype=torch.int32)


class LinearQuant(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bit_prec: int,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        bias: bool,
        weight_noise_func: Optional[Union[Callable, str]],
        weight_noise_values: Optional[Dict[str, float]],
        weight_noise_start_epoch: Optional[int],
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), requires_grad=True)
        else:
            self.bias = None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec,
            dtype=self.weight_quant_dtype,
            method=self.weight_quant_method,
        )
        self.weight_noise_func = weight_noise_func
        self.weight_noise_start_epoch = weight_noise_start_epoch
        self.weight_noise_values = weight_noise_values

    def forward(self, tensor: torch.Tensor):
        weight = self.weight_quantizer(self.weight)
        if self.weight_noise_func is not None and (self.weight_noise_start_epoch is None or not self.training or self.weight_noise_start_epoch >= get_run_ctx().epoch):
            for i in range(self.weight_bit_prec-1):
                mean = 2 * (-self.weight_quantizer.zero_point).expand(self.weight.size()).to(weight.device).to(torch.float32)
                std = (torch.tensor(self.weight_noise_values["dev"]) * (2 ** i)).expand(self.weight.size()).to(weight.device).to(torch.float32)
                std = numpy.sqrt(2) * std  # this is sqrt(std ^ 2 + std ^ 2)
                noise = self.weight_noise_func(mean=mean, std=std).to(weight.device) * self.weight_quantizer.scale
                weight = weight + noise
        lin = F.linear(tensor, weight, self.bias)
        return lin


class Conv1dQuant(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        weight_bit_prec: int,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        bias: bool,
        stride: int,
        padding: Union[str, int],
        dilation: int,
        groups: int,
        weight_noise_func: Optional[Union[Callable, str]],
        weight_noise_values: Optional[Dict[str, float]],
        weight_noise_start_epoch: Optional[int],
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec,
            dtype=self.weight_quant_dtype,
            method=self.weight_quant_method,
        )
        self.weight_noise_func = weight_noise_func
        self.weight_noise_start_epoch = weight_noise_start_epoch
        self.weight_noise_values = weight_noise_values

    def forward(self, tensor: torch.Tensor):
        weight = self.weight_quantizer(self.weight)
        if self.weight_noise_func is not None and (self.weight_noise_start_epoch is None or not self.training or self.weight_noise_start_epoch >= get_run_ctx().epoch):
            for i in range(self.weight_bit_prec-1):
                mean = 2 * (-self.weight_quantizer.zero_point).expand(self.weight.size()).to(weight.device).to(torch.float32)
                std = (torch.tensor(self.weight_noise_values["dev"]) * (2 ** i)).expand(self.weight.size()).to(weight.device).to(torch.float32)
                std = numpy.sqrt(2) * std  # this is sqrt(std ^ 2 + std ^ 2)
                noise = self.weight_noise_func(mean=mean, std=std).to(weight.device) * self.weight_quantizer.scale
                weight = weight + noise
        result = F.conv1d(
            tensor, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return result


class QuantizedMultiheadAttention(nn.Module):
    """
        Conformer multi-headed self-attention module supporting
            - self-attention with relative positional encoding proposed by Shaw et al. (cf. https://arxiv.org/abs/1803.02155)
                * learnable_pos_emb = True
                * with_pos_bias = False
                * with_linear_pos = False
                * separate_pos_emb_per_head = False (RETURNN default)
                * with_bias = False (RETURNN default)
            - and self-attention with Transformer-XL style relative PE by Dai et al.
                (cf. https://arxiv.org/abs/1901.02860, https://github.com/kimiyoung/transformer-xl/blob/master/pytorch/mem_transformer.py,
                     https://github.com/espnet/espnet/blob/master/espnet2/asr_transducer/encoder/modules/attention.py#L9)
                * learnable_pos_emb = False
                * with_pos_bias = True
                * with_linear_pos = False (paper implementation) / with_linear_pos = True (ESPnet default)
                * separate_pos_emb_per_head = False (paper implementation) / separate_pos_emb_per_head = True (ESPnet default)
                * with_bias = False (paper implementation) / with_bias = True (ESPnet default)
        """

    def __init__(self, cfg: QuantizedConformerMHSARelPosV1Config):

        super().__init__()

        self.layernorm = nn.LayerNorm(cfg.input_dim)

        self.embed_dim = cfg.input_dim
        self.num_heads = cfg.num_att_heads
        self.embed_dim_per_head = self.embed_dim // self.num_heads

        self.learnable_pos_emb = cfg.learnable_pos_emb
        self.rel_pos_clip = cfg.rel_pos_clip
        self.separate_pos_emb_per_head = cfg.separate_pos_emb_per_head
        self.with_pos_bias = cfg.with_pos_bias
        self.pos_emb_dropout = nn.Dropout(cfg.pos_emb_dropout)
        self.weight_quant_dtype = cfg.weight_quant_dtype
        self.weight_quant_method = cfg.weight_quant_method
        self.weight_noise_start_epoch = cfg.weight_noise_start_epoch
        self.weight_noise_values = cfg.weight_noise_values
        self.weight_noise_func = cfg.weight_noise_func
        self.bit_prec_dot = cfg.bit_prec_dot
        self.activation_quant_dtype = cfg.activation_quant_dtype
        self.activation_quant_method = cfg.activation_quant_method
        self.dot_quant_dtype = cfg.dot_quant_dtype
        self.dot_quant_method = cfg.dot_quant_method
        self.Av_quant_dtype = cfg.Av_quant_dtype
        self.Av_quant_method = cfg.Av_quant_method
        self.converter_hardware_settings = cfg.converter_hardware_settings


        assert not self.learnable_pos_emb or self.rel_pos_clip

        self.att_weights_dropout = nn.Dropout(cfg.att_weights_dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"
        from torch_memristor.memristor_modules import TiledMemristorLinear

        # projection matrices
        self.qkv_proj = TiledMemristorLinear(
            in_features=self.embed_dim,
            out_features=3 * self.embed_dim,
            weight_precision=cfg.bit_prec_W_i if not cfg.bit_prec_W_i == 1.5 else 2,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        self.in_proj_in_quant = nn.Identity()

        self.in_proj_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )
        self.q_quantizer = ActivationQuantizer(
            self.bit_prec_dot,
            self.dot_quant_dtype,
            self.dot_quant_method,
            channel_axis=None if self.dot_quant_method == "per_tensor" else 3,
            moving_avrg=cfg.moving_average,
        )
        self.k_quantizer = ActivationQuantizer(
            self.bit_prec_dot,
            self.dot_quant_dtype,
            self.dot_quant_method,
            channel_axis=None if self.dot_quant_method == "per_tensor" else 2,
            moving_avrg=cfg.moving_average,
        )

        self.out_proj = TiledMemristorLinear(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            weight_precision=cfg.bit_prec_W_o if not cfg.bit_prec_W_o == 1.5 else 2,
            converter_hardware_settings=cfg.converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
        self.out_proj_in_quant = nn.Identity()

        self.out_proj_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.register_parameter("rel_pos_embeddings", None)
        self.register_parameter("pos_bias_u", None)
        self.register_parameter("pos_bias_v", None)

        self.pos_emb_dim = (
            self.embed_dim if cfg.with_linear_pos or cfg.separate_pos_emb_per_head else self.embed_dim_per_head
        )
        if self.learnable_pos_emb:
            self.rel_pos_embeddings = nn.parameter.Parameter(torch.empty(self.rel_pos_clip * 2 + 1, self.pos_emb_dim))
        if cfg.with_linear_pos:
            self.linear_pos = LinearQuant(
                in_features=self.pos_emb_dim,
                out_features=self.embed_dim if cfg.separate_pos_emb_per_head else self.embed_dim_per_head,
                weight_bit_prec=cfg.bit_prec_learn_emb,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                bias=False,
                weight_noise_func=self.weight_noise_func,
                weight_noise_start_epoch=self.weight_noise_start_epoch,
                weight_noise_values=self.weight_noise_values,
            )
            self.learn_emb_in_quant = ActivationQuantizer(
                bit_precision=cfg.activation_bit_prec,
                dtype=cfg.activation_quant_dtype,
                method=cfg.activation_quant_method,
                channel_axis=1,
                moving_avrg=cfg.moving_average,
            )

            self.learn_emb_out_quant = ActivationQuantizer(
                bit_precision=cfg.activation_bit_prec,
                dtype=cfg.activation_quant_dtype,
                method=cfg.activation_quant_method,
                channel_axis=1,
                moving_avrg=cfg.moving_average,
            )
        else:
            self.linear_pos = nn.Identity()

        if self.with_pos_bias:
            self.pos_bias_u = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))
            self.pos_bias_v = nn.parameter.Parameter(torch.empty(self.num_heads, self.embed_dim_per_head))

        self.dropout = BroadcastDropout(cfg.dropout, dropout_broadcast_axes=cfg.dropout_broadcast_axes)
        self.cfg = cfg

        self._reset_parameters()

    def _reset_parameters(self):
        if self.learnable_pos_emb:
            nn.init.xavier_normal_(self.rel_pos_embeddings)
        if self.with_pos_bias:
            # init taken from espnet default
            nn.init.xavier_uniform_(self.pos_bias_u)
            nn.init.xavier_uniform_(self.pos_bias_v)

    def forward(self, input_tensor: torch.Tensor, sequence_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply layer norm and multi-head self attention and dropout

        :param input_tensor: Input to the self attention of shape (B, T, F)
        :param sequence_mask: bool mask of shape (B, T), True signals within sequence, False outside
        """
        output_tensor = self.layernorm(input_tensor)  # [B, T, F]
        time_dim_size = output_tensor.shape[1]
        batch_dim_size = output_tensor.shape[0]

        # attention mask
        # T: query seq. length, T' key/value seg length; T = T' if same input tensor
        inv_sequence_mask = compat.logical_not(sequence_mask)  # [B, T']
        mask = (
            torch.zeros_like(inv_sequence_mask, dtype=input_tensor.dtype)
            .masked_fill(inv_sequence_mask, float("-inf"))
            .reshape(batch_dim_size, 1, 1, time_dim_size)
        )  # [B, 1, 1, T']

        # query, key and value sequences
        output_tensor = self.in_proj_in_quant(output_tensor)
        query_seq, key_seq, value_seq = self.qkv_proj(output_tensor).chunk(3, dim=-1)  # [B, T, #heads * F']

        q = query_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, #heads, F']
        k = key_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T', #heads, F']
        q = self.q_quantizer(q)
        k = self.k_quantizer(k)


        if self.learnable_pos_emb:
            pos_seq_q = torch.arange(time_dim_size, device=input_tensor.device)
            pos_seq_k = torch.arange(time_dim_size, device=input_tensor.device)

            distance_mat = pos_seq_k[None, :] - pos_seq_q[:, None]
            distance_mat_clipped = torch.clamp(distance_mat, -self.rel_pos_clip, self.rel_pos_clip)

            final_mat = distance_mat_clipped + self.rel_pos_clip

            rel_pos_embeddings = self.rel_pos_embeddings[final_mat]  # [T, T', pos_emb_dim]
        else:
            rel_pos_embeddings = self._sinusoidal_pe(
                torch.arange(time_dim_size - 1, -time_dim_size, -1, device=input_tensor.device, dtype=torch.float32),
                self.pos_emb_dim,
            ).view(
                1, 2 * time_dim_size - 1, self.pos_emb_dim
            )  # [1, T+T'-1, pos_emb_dim]

        # dropout relative positional embeddings
        rel_pos_embeddings = self.pos_emb_dropout(
            rel_pos_embeddings
        )  # [T, T', pos_emb_dim] or [1, T+T'-1, pos_emb_dim]
        rel_pos_embeddings = rel_pos_embeddings.unsqueeze(2)  # [T, T', 1, pos_emb_dim] or [1, T+T'-1, 1, pos_emb_dim]

        # linear transformation or identity
        if self.cfg.with_linear_pos:
            rel_pos_embeddings = self.learn_emb_in_quant(rel_pos_embeddings)
        rel_pos_embeddings = self.linear_pos(rel_pos_embeddings)  # [T, T', 1, F'|F] or [1, T+T'-1, 1, F'|F]

        if self.cfg.with_linear_pos:
            rel_pos_embeddings = self.learn_emb_out_quant(rel_pos_embeddings)

        if self.separate_pos_emb_per_head:
            rel_pos_embeddings = rel_pos_embeddings.squeeze(2).reshape(
                *rel_pos_embeddings.shape[:2], -1, self.embed_dim_per_head
            )  # [T, T', #heads, F'] or [1, T+T'-1, #heads, F']

        q_with_bias_u = q + self.pos_bias_u if self.with_pos_bias else q  # [B, T, #heads, F']
        q_with_bias_v = q + self.pos_bias_v if self.with_pos_bias else q

        # attention matrix a and c
        attn_ac = torch.einsum("bihf, bjhf -> bhij", q_with_bias_u, k)  # [B, #heads, T, T']

        # attention matrix b and d
        attn_bd = torch.einsum(
            "bihf, ijhf -> bhij", q_with_bias_v, rel_pos_embeddings
        )  # [B, #heads, T, T'] or [B, #heads, T, T+T'+1]

        if not self.learnable_pos_emb:
            attn_bd = self._rel_shift_bhij(attn_bd, k_len=time_dim_size)  # [B, #heads, T, T']

        attn = attn_ac + attn_bd + mask  # [B, #heads, T, T']
        attn_scaled = attn * (math.sqrt(1.0 / float(self.embed_dim_per_head)))  # [B, #heads, T, T']

        # softmax and dropout
        attn_output_weights = self.att_weights_dropout(F.softmax(attn_scaled, dim=-1))  # [B, #heads, T, T']

        # sequence of weighted sums over value sequence
        v = value_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, H, F']
        attn_output = torch.einsum("bhij, bjhf -> bihf", attn_output_weights, v).reshape(
            batch_dim_size, -1, self.embed_dim
        )

        attn_output = self.out_proj_in_quant(attn_output)
        output_tensor = self.out_proj(attn_output)
        output_tensor = self.out_proj_out_quant(output_tensor)

        output_tensor = self.dropout(output_tensor)

        return output_tensor  # [B,T,F]

    @staticmethod
    def _rel_shift_bhij(x, k_len=None):
        """
        :param x: input tensor of shape (B, H, T, L) to apply left shift
        :k_len: length of the key squence
        """
        x_shape = x.shape

        x = torch.nn.functional.pad(x, (1, 0))  # [B, H, T, L+1]
        x = x.reshape(x_shape[0], x_shape[1], x_shape[3] + 1, x_shape[2])  # [B, H, L+1, T]
        x = x[:, :, 1:]  # [B, H, L, T]
        x = x.reshape(x_shape)  # [B, H, T, L]]

        return x[:, :, :, :k_len] if k_len else x  # [B, H, T, T']

    @staticmethod
    def _sinusoidal_pe(pos_seq: torch.Tensor, embed_dim: int):
        """
        :param pos_seq: 1-D position sequence for which to compute embeddings
        :param embed_dim: embedding dimension
        """
        inv_freq = 1 / (10000 ** (torch.arange(0.0, embed_dim, 2.0, device=pos_seq.device) / embed_dim))

        sinusoid_input = torch.outer(pos_seq, inv_freq)

        pos_emb = torch.zeros(pos_seq.shape[0], embed_dim, device=pos_seq.device)

        pos_emb[:, 0::2] = sinusoid_input.sin()
        pos_emb[:, 1::2] = sinusoid_input.cos()

        return pos_emb

    def prep_quant(self):
        self.qkv_proj.initialized = True
        self.out_proj.initialized = True
