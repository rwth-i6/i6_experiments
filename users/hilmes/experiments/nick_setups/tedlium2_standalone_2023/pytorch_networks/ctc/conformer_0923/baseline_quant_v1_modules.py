import torch
from torch import nn
import torch.ao.quantization as torch_quant
import torch.nn.functional as F
from typing import Optional
from .baseline_quant_v1_cfg import QuantizedMultiheadAttentionV1Config
import math
from returnn.torch.context import get_run_ctx

def get_quantization_range_from_bit_precision(bits, dtype):

    if dtype == torch.qint8:
        quant_min = -(2**(bits-1))
        quant_max = (2**(bits-1))-1

    elif dtype == torch.quint8:
        quant_min = 0
        quant_max = (2**bits)-1

    else:
        raise ValueError(f'Unrecognized dtype {dtype}')

    return quant_min, quant_max


class WeightQuantizer(nn.Module):
    def __init__(self, bit_precision: int, dtype: torch.dtype, method: str, reduce_range: bool = False):
        super().__init__()

        self.quant_min, self.quant_max = get_quantization_range_from_bit_precision(bit_precision, dtype)
        self.dtype = dtype
        self.quant_fn, self.observer = None, None
        self.quant_fn, self.observer = self.__get_quant_fn_and_observer_for_method(method)
        self.reduce_rage = reduce_range
        self.scale = None
        self.zero_point = None

    def __get_quant_fn_and_observer_for_method(self, method):
        if self.quant_fn is not None and self.observer is not None:
            return self.quant_fn, self.observer
        if method == 'per_tensor':
            quant_fn = torch.fake_quantize_per_tensor_affine
            observer = torch_quant.observer.MinMaxObserver(
                quant_min=self.quant_min,
                quant_max=self.quant_max,
                dtype=self.dtype,
                reduce_range=self.reduce_rage
            )
        else:
            raise ValueError(f'Unknown quantization method: {method}!')

        return quant_fn, observer

    def forward(self, tensor: torch.Tensor):
        if self.training:
            # TODO This module does not do anything in training
            return tensor
            #self.observer.reset_min_max_vals()
        if not self.apply_calibration:
            print("Calculating Min Max Values")
            tensor = self.observer(tensor)
        if self.iterative_quant or self.apply_calibration:
            # TODO: self.iterative_quant and self.apply_calibration needs to be the proper variation
            # TODO: How to check state?
            # TODO: Ich glaube ein großer Unterschied ist, dass hier quantisiert wird nachdem man die values gesehen hat
            # und nicht erst ganz am Ende. Heißt jeder Batch wird iterativ quantisiert
            self.set_scale_and_zp()
            assert self.scale and self.zero_point
            print("Applying Quantized values")
            tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
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
        self.quant_fn, self.observer = None, None
        self.quant_fn, self.observer, self.base_observer_args = self.__get_quant_fn_and_observer_for_method(method)
        self.channel_axis = channel_axis
        self.moving_avrg = moving_avrg
        self.reduce_range = reduce_range
        self.zero_point = None
        self.scale = None

    def __get_quant_fn_and_observer_for_method(self, method):
        if all(x is not None for x in [self.quant_fn, self.base_observer_args, self.observer]):
            return self.quant_fn, self.base_observer_args, self.observer
        if method == 'per_tensor':
            quant_fn = torch.fake_quantize_per_tensor_affine
            base_observer_args = [self.quant_min, self.quant_max]
            if self.moving_avrg:
                observer = torch_quant.observer.MovingAverageMinMaxObserver(
                    averaging_constant=self.moving_avrg,
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range
                )
            else:
                observer = torch_quant.observer.MinMaxObserver(
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_rage
                )
        elif method == 'per_channel':
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
                    reduce_range=self.reduce_range
                )
            else:
                observer = torch_quant.observer.PerChannelMinMaxObserver(
                    quant_min=self.quant_min,
                    quant_max=self.quant_max,
                    dtype=self.dtype,
                    reduce_range=self.reduce_range,
                    ch_axis=self.channel_axis
                )
        else:
            raise ValueError(f'Unknown quantization method: {method}!')

        return quant_fn, observer, base_observer_args


    def forward(self, tensor: torch.Tensor):
        # TODO
        # tensor = self.observer(tensor)
        # scale, zero_point = self.observer.calculate_qparams()
        # tensor = self.quant_fn(tensor, scale, zero_point, *self.base_observer_args)
        # return tensor
        if self.training:
            # TODO This module does not do anything in training
            return tensor
            # self.observer.reset_min_max_vals()
        if not self.apply_calibration:
            print("Calculating Min Max Values")
            tensor = self.observer(tensor)
        # TODO: remove, only for debugging rn
        if self.apply_calibration:
            assert self.apply_calibration is True
        if self.iterative_quant:
            assert self.iterative_quant is True
        if self.iterative_quant or self.apply_calibration:
            # TODO: How to check state?
            # TODO: Ich glaube ein großer Unterschied ist, dass hier quantisiert wird nachdem man die values gesehen hat
            # und nicht erst ganz am Ende. Heißt jeder Batch wird iterativ quantisiert
            self.set_scale_and_zp()
            assert self.scale and self.zero_point, "Need to calibrate before applying quant, disable apply_calibration"
            print("Applying Quantized values")
            tensor = self.quant_fn(tensor, self.scale, self.zero_point, self.quant_min, self.quant_max)
        return tensor

    def set_scale_and_zp(self):
        assert self.observer is not None
        self.scale, self.zero_point = self.observer.calculate_qparams()


class LinearQuant(nn.Module):

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bit_prec: int,
            weight_quant_dtype: torch.dtype,
            weight_quant_method: str,
            activation_quant_dtype: torch.dtype,
            activation_quant_method: str,
            moving_average: Optional[float],  # default if enabled should be 0.01, if set enables moving average
            bias: bool
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.empty((in_features, out_features)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), requires_grad=True)
        self.bit_prec = bit_prec

        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(self.bit_prec, self.weight_quant_dtype, self.weight_quant_method)

        self.activation_quant_dtype = activation_quant_dtype
        self.activation_quant_method = activation_quant_method
        self.activation_quantizer = ActivationQuantizer(
            bit_precision=self.bit_prec,
            dtype=self.activation_quant_dtype,
            method=self.activation_quant_method,
            channel_axis=2,
            moving_avrg=moving_average)

    def forward(self, tensor: torch.Tensor):
        return F.linear(self.activation_quantizer(tensor), self.weight_quantizer(self.weight), self.bias)

# TODO: Check if this is all correct
class QuantizedMultiheadAttention(nn.Module):
    def __init__(
            self,
            cfg: QuantizedMultiheadAttentionV1Config
    ):
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

        self.W_q = self._create_linear_layer(cfg.bit_prec_W_q)
        self.W_k = self._create_linear_layer(cfg.bit_prec_W_k)
        self.W_v = self._create_linear_layer(cfg.bit_prec_W_v)
        self.W_o = self._create_linear_layer(cfg.bit_prec_W_o)

        if self.bit_prec_dot < 16:
            self.q_quantizer = ActivationQuantizer(
                self.bit_prec_dot,
                self.dot_quant_dtype,
                self.dot_quant_method,
                channel_axis=None if self.dot_quant_method == "per_tensor" else 3,
                moving_avrg=cfg.moving_average
            )
            self.k_quantizer = ActivationQuantizer(
                self.bit_prec_dot,
                self.dot_quant_dtype,
                self.dot_quant_method,
                channel_axis=None if self.dot_quant_method == "per_tensor" else 2,
                moving_avrg=cfg.moving_average
            )

        if self.bit_prec_Av < 16:
            self.a_quantizer = WeightQuantizer(
                self.bit_prec_Av,
                self.Av_quant_dtype,
                self.Av_quant_method
            )
            self.v_quantizer = ActivationQuantizer(
                self.bit_prec_Av,
                self.Av_quant_dtype,
                self.Av_quant_method,
                moving_avrg=cfg.moving_average,
                channel_axis=None if self.dot_quant_method == "per_tensor" else NotImplementedError,
            )
        self.norm = math.sqrt(self.input_dim)
        self.softmax = nn.Softmax(-1)
        self.dropout = nn.Dropout(cfg.att_weights_dropout)

    def _create_linear_layer(self, bits):
        if bits < 16:  # TODO: why 16?
            return LinearQuant(
                in_features=self.input_dim,
                out_features=self.input_dim,
                bit_prec=bits,
                weight_quant_dtype=self.weight_quant_dtype,
                weight_quant_method=self.weight_quant_method,
                activation_quant_dtype=self.activation_quant_dtype,
                activation_quant_method=self.activation_quant_method,
                moving_average=self.cfg.moving_average,
                bias=False,  # TODO: This has to be false right?
            )
        else:
            return nn.Linear(self.input_dim, self.input_dim, bias=False)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: Optional[torch.Tensor] = None):

        batch_dim = query.shape[0]

        query = self.W_q(query)
        key = self.W_k(key)
        value = self.W_v(value)

        query = query.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        key = key.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']
        value = value.view(batch_dim, -1, self.num_att_heads, self.dim_heads)  # [B, T, D//H, D']

        query = torch.transpose(query, 1, 2)  # [B, D//H, T, D']
        key = torch.transpose(key, 1, 2)  # [B, D//H, T, D']
        value = torch.transpose(value, 1, 2)  # [B, D//H, T, D']

        key = torch.transpose(key, -2, -1)  # [B, D//H, D', T]

        if self.bits_dot < 16:
            query = self.q_quantizer(query)
            key = self.k_quantizer(key)

        dot = torch.matmul(query, key)  # [B, D//H, T, T]
        dot = dot / self.norm
        if mask is not None:
            dot = dot.masked_fill(dot, -float('inf'))
        alpha = self.softmax(dot)
        alpha = self.dropout(alpha)

        if self.bits_Av < 16:
            alpha = self.a_quantizer(alpha)
            value = self.v_quantizer(value)

        att_out = torch.matmul(alpha, value)  # [B, D//H, T, D']
        att_out = torch.transpose(att_out, 1, 2)  # [B, D//H, T, D']
        att_out = att_out.reshape(batch_dim, -1, self.input_dim)  # [B, T, D]
        att_out = self.W_o(att_out)

        return att_out, alpha
