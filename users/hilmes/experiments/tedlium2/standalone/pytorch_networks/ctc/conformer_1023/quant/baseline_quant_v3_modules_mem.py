"""
Fixes mhsa norm
mem adds memristor bit correction
v3 adds convolution as quantized
"""
import copy

import torch
from torch import nn
from torch.nn import init
import torch.ao.quantization as torch_quant
import torch.nn.functional as F
from typing import Optional, Union
from .baseline_quant_v1_cfg import QuantizedMultiheadAttentionV1Config
import math
from returnn.torch.context import get_run_ctx
from torch.ao.quantization.utils import check_min_max_valid


def get_quantization_range_from_bit_precision(bits, dtype):

    if bits == 1.5:
        quant_min = -1
        quant_max = 1
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
    def __init__(self, bit_precision: int, dtype: torch.dtype, method: str, reduce_range: bool = False):
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
            # This module does not do anything in training
            return tensor
        if not get_run_ctx().apply_quant:
            tensor = self.observer(tensor)
        if get_run_ctx().iterative_quant or get_run_ctx().apply_quant:
            # und nicht erst ganz am Ende. Heißt jeder Batch wird iterativ quantisiert
            self.set_scale_and_zp()
            assert self.scale is not None and self.zero_point is not None
            tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        assert check_min_max_valid(self.observer.min_val, self.observer.max_val), "Need to init observer first"
        self.scale, self.zero_point = self.observer.calculate_qparams()


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
            # This module does not do anything in training
            return tensor
            # self.observer.reset_min_max_vals()
        if get_run_ctx().apply_quant is not True:
            tensor = self.observer(tensor)
        if get_run_ctx().iterative_quant is True or get_run_ctx().apply_quant is True:
            self.set_scale_and_zp()
            assert (
                self.scale is not None and self.zero_point is not None
            ), "Need to calibrate before applying quant, disable apply_calibration"
            old_tensor = copy.deepcopy(tensor)
            tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
            # This 0 case should only happen when the model is broken anyways like for 4 bit
            # TODO it seems for linear out this might be the case
            # assert not torch.equal(old_tensor, tensor) or torch.sum(old_tensor) == torch.sum(tensor) == 0, (tensor[0,0,0], old_tensor[0 , 0 ,0], self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        assert check_min_max_valid(self.observer.min_val, self.observer.max_val), "Need to init observer first"
        self.scale, self.zero_point = self.observer.calculate_qparams()


class LinearQuant(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        weight_bit_prec: int,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        activation_bit_prec: int,
        activation_quant_dtype: torch.dtype,
        activation_quant_method: str,
        moving_average: Optional[float],  # default if enabled should be 0.01, if set enables moving average
        bias: bool,
        quant_output: bool,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((out_features, in_features)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), requires_grad=True)

        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec, dtype=self.weight_quant_dtype, method=self.weight_quant_method
        )

        self.activation_bit_prec = activation_bit_prec
        self.activation_quant_dtype = activation_quant_dtype
        self.activation_quant_method = activation_quant_method
        self.activation_quantizer = ActivationQuantizer(
            bit_precision=self.activation_bit_prec,
            dtype=self.activation_quant_dtype,
            method=self.activation_quant_method,
            channel_axis=2,
            moving_avrg=moving_average,
        )

        self.quant_output = quant_output
        if self.quant_output:
            self.output_quantizer = ActivationQuantizer(
                bit_precision=self.activation_bit_prec,
                dtype=self.activation_quant_dtype,
                method=self.activation_quant_method,
                channel_axis=2,
                moving_avrg=moving_average,
            )

    def forward(self, tensor: torch.Tensor):
        lin = F.linear(self.activation_quantizer(tensor), self.weight_quantizer(self.weight), self.bias)
        if self.quant_output:
            return self.output_quantizer(lin)
        else:
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
        activation_bit_prec: int,
        activation_quant_dtype: torch.dtype,
        moving_average: Optional[float],  # default if enabled should be 0.01, if set enables moving average
        activation_quant_method: str,
        padding_mode: str = "zeros",  # TODO: refine this type
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

        self.activation_bit_prec = activation_bit_prec
        self.activation_quant_dtype = activation_quant_dtype
        self.activation_quant_method = activation_quant_method
        self.activation_quantizer = ActivationQuantizer(
            bit_precision=self.activation_bit_prec,
            dtype=self.activation_quant_dtype,
            method=self.activation_quant_method,
            channel_axis=2,
            moving_avrg=moving_average,
        )

    def forward(self, tensor: torch.Tensor):
        result = F.conv1d(
            self.activation_quantizer(tensor),
            self.weight_quantizer(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result


class QuantizedMultiheadAttention(nn.Module):
    def __init__(self, cfg: QuantizedMultiheadAttentionV1Config):
        super().__init__()
        self.cfg = cfg
        self.num_att_heads = cfg.num_att_heads
        self.input_dim = cfg.input_dim
        self.dim_heads = self.input_dim // self.num_att_heads

        self.bit_prec_dot = cfg.bit_prec_dot
        self.bit_prec_Av = cfg.bit_prec_A_v
        self.weight_quant_dtype = cfg.weight_quant_dtype
        self.weight_quant_method = cfg.weight_quant_method
        self.activation_quant_dtype = cfg.activation_quant_dtype
        self.activation_quant_method = cfg.activation_quant_method
        self.dot_quant_dtype = cfg.dot_quant_dtype
        self.dot_quant_method = cfg.dot_quant_method
        self.Av_quant_dtype = cfg.Av_quant_dtype
        self.Av_quant_method = cfg.Av_quant_method
        self.linear_quant_output = cfg.linear_quant_output

        self.out_proj = self._create_linear_layer(
            weight_bits=cfg.bit_prec_W_o,
            act_bits=cfg.activation_bit_prec,
        )

        # For some reason pytorch saves the in_proj_weight and bias in this format not with . so we need to adjust
        self.in_proj = self._create_linear_layer(
            weight_bits=cfg.bit_prec_W_q, act_bits=cfg.activation_bit_prec, output_dim=3 * self.input_dim
        )
        self.register_parameter("in_proj_weight", self.in_proj.weight)
        self.register_parameter("in_proj_bias", self.in_proj.bias)

        if self.bit_prec_dot < 16:
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

        if self.bit_prec_Av < 16:
            self.a_quantizer = WeightQuantizer(self.bit_prec_Av, self.Av_quant_dtype, self.Av_quant_method)
            self.v_quantizer = ActivationQuantizer(
                self.bit_prec_Av,
                self.Av_quant_dtype,
                self.Av_quant_method,
                moving_avrg=cfg.moving_average,
                channel_axis=None if self.dot_quant_method == "per_tensor" else NotImplementedError,
            )
        self.norm = math.sqrt(self.dim_heads)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(cfg.att_weights_dropout)

    def _create_linear_layer(self, weight_bits, act_bits, output_dim=None):
        return LinearQuant(
            in_features=self.input_dim,
            out_features=output_dim or self.input_dim,
            weight_bit_prec=weight_bits,
            weight_quant_dtype=self.weight_quant_dtype,
            weight_quant_method=self.weight_quant_method,
            activation_bit_prec=act_bits,
            activation_quant_dtype=self.activation_quant_dtype,
            activation_quant_method=self.activation_quant_method,
            moving_average=self.cfg.moving_average,
            bias=True,
            quant_output=self.linear_quant_output,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):

        batch_dim = query.shape[0]

        # query = self.W_q(query)
        # key = self.W_k(key)
        # value = self.W_v(value)
        assert query is value is key, "currently only this case is implemented"

        x = self.in_proj(query)
        hidden_dim = query.size(-1)
        query, key, value = x.unflatten(-1, (3, hidden_dim)).unsqueeze(0).transpose(0, -2).squeeze(-2).contiguous()

        query = query.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        key = key.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        value = value.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']

        query = torch.transpose(query, 1, 2)  # [B, D//H, T, D']
        key = torch.transpose(key, 1, 2)  # [B, D//H, T, D']
        value = torch.transpose(value, 1, 2)  # [B, D//H, T, D']

        key = torch.transpose(key, -2, -1)  # [B, D//H, D', T]

        if self.bit_prec_dot < 16:
            query = self.q_quantizer(query)
            key = self.k_quantizer(key)

        dot = torch.matmul(query, key)  # [B, D//H, T, T]
        dot = dot / self.norm
        if mask is not None:
            mask = mask.view(batch_dim, 1, 1, mask.size(1))
            dot = dot.masked_fill(mask, -float("inf"))
        alpha = self.softmax(dot)
        # alpha = self.dropout(alpha)

        if self.bit_prec_Av < 16:
            alpha = self.a_quantizer(alpha)
            value = self.v_quantizer(value)

        att_out = torch.matmul(alpha, value)  # [B, D//H, T, D']
        att_out = torch.transpose(att_out, 1, 2)  # [B, D//H, T, D']
        att_out = att_out.reshape(batch_dim, -1, self.input_dim)  # [B, T, D]
        att_out = self.out_proj(att_out)

        return att_out, alpha
