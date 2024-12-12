"""
Fixes mhsa norm
"""

import torch
from torch import nn
from torch.nn import init
import torch.ao.quantization as torch_quant
import torch.nn.functional as F
from typing import Optional, Union
from .memristor_v1_cfg import QuantizedMultiheadAttentionV4Config
import math
from torch.ao.quantization.utils import check_min_max_valid


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
        tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
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

    def forward(self, tensor: torch.Tensor):
        lin = F.linear(tensor, self.weight_quantizer(self.weight), self.bias)
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

        self.weight = nn.Parameter(torch.empty(in_channels, out_channels // groups, kernel_size), requires_grad=True)
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

    def forward(self, tensor: torch.Tensor):
        result = F.conv1d(
            tensor, self.weight_quantizer(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return result


class QuantizedMultiheadAttention(nn.Module):
    def __init__(
        self,
        cfg: QuantizedMultiheadAttentionV4Config,
    ):
        super().__init__()
        self.quant_in_linear = cfg.quant_in_linear
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
        self.converter_hardware_settings = cfg.converter_hardware_settings

        self.out_proj = self._create_linear_layer(
            weight_bits=cfg.bit_prec_W_o,
        )
        self.out_proj_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.out_proj_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        # For some reason pytorch saves the in_proj_weight and bias in this format not with . so we need to adjust
        self.in_proj = self._create_linear_layer(weight_bits=cfg.bit_prec_W_q, output_dim=3 * self.input_dim)
        self.in_proj_in_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
        )

        self.in_proj_out_quant = ActivationQuantizer(
            bit_precision=cfg.activation_bit_prec,
            dtype=cfg.activation_quant_dtype,
            method=cfg.activation_quant_method,
            channel_axis=1,
            moving_avrg=cfg.moving_average,
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

    def _create_linear_layer(self, weight_bits, output_dim=None):
        return LinearQuant(
            in_features=self.input_dim,
            out_features=output_dim or self.input_dim,
            weight_bit_prec=weight_bits,
            weight_quant_dtype=self.weight_quant_dtype,
            weight_quant_method=self.weight_quant_method,
            bias=True,
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):

        batch_dim = query.shape[0]

        # query = self.W_q(query)
        # key = self.W_k(key)
        # value = self.W_v(value)
        assert query is value is key, "currently only this case is implemented"

        if self.quant_in_linear is True:  # TODO: I guess this is False in forward? Maybe just set to Identity?
            query = self.in_proj_in_quant(query)
        x = self.in_proj(query)
        if self.quant_in_linear is True:
            x = self.in_proj_out_quant(x)
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
        if self.quant_in_linear is True:
            att_out = self.out_proj_in_quant(att_out)
        att_out = self.out_proj(att_out)
        if self.quant_in_linear is True:
            att_out = self.out_proj_out_quant(att_out)

        return att_out, alpha

    def prep_quant(self):
        from torch_memristor.memristor_modules import MemristorLinear

        self.out_proj.weight_quantizer.set_scale_and_zp()
        self.out_proj_in_quant.set_scale_and_zp()
        mem_lin = MemristorLinear(
            in_features=self.out_proj.in_features,
            out_features=self.out_proj.out_features,
            weight_precision=self.out_proj.weight_bit_prec,
            converter_hardware_settings=self.converter_hardware_settings,
        )
        mem_lin.init_from_linear_quant(activation_quant=self.out_proj_in_quant, linear_quant=self.out_proj)
        self.out_proj = mem_lin
        self.out_proj_in_quant = nn.Identity()

        self.in_proj.weight_quantizer.set_scale_and_zp()
        self.in_proj_in_quant.set_scale_and_zp()
        mem_lin = MemristorLinear(
            in_features=self.in_proj.in_features,
            out_features=self.in_proj.out_features,
            weight_precision=self.in_proj.weight_bit_prec,
            converter_hardware_settings=self.converter_hardware_settings,
        )
        mem_lin.init_from_linear_quant(activation_quant=self.in_proj_in_quant, linear_quant=self.in_proj)
        self.in_proj = mem_lin
        self.in_proj_in_quant = nn.Identity()
