"""
Fixes mhsa norm
v10 fixes memristor init
v11 adds option for separate dac for pos encs
v12 adds option for layerwise
v13 adds weight dropout
v14 adds weight sparsity / pruning
"""

import torch
from torch import nn
from torch.nn import init
import torch.ao.quantization as torch_quant
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from .memristor_v16_dynmic_prec_cfg import (
    QuantizedConformerMHSARelPosV1Config,
    WeightPruningConfig,
    WeightNoiseConfig,
    WeightPrecSpec,
    SubMatrixPrecision,
)
import math
from torch.ao.quantization.utils import check_min_max_valid

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

        self.bit_precision = bit_precision
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


class TiledWeightQuantizer(nn.Module):
    """Per-tile weight fake-quantization for mixed-precision matrices.

    Each 128x128 tile of the weight matrix W [out_features, in_features] gets its own
    WeightQuantizer (own observer, scale, zero point) with the bit precision given by the
    SubMatrixPrecision grid. The forward pass fake-quantizes every tile separately and
    reassembles the full weight with torch.cat, so gradients flow per tile via the usual STE.
    """

    is_tiled = True

    def __init__(self, sub_matrix_precision: SubMatrixPrecision, dtype: torch.dtype, method: str):
        super().__init__()
        self.sub_matrix_precision = sub_matrix_precision
        self.dtype = dtype
        self.method = method
        self.quantizers = nn.ModuleList(
            nn.ModuleList(WeightQuantizer(bit_precision=prec, dtype=dtype, method=method) for prec in row)
            for row in sub_matrix_precision.grid
        )

    def get_tile_quantizer(self, r: int, c: int) -> WeightQuantizer:
        return self.quantizers[r][c]

    def forward(self, tensor: torch.Tensor):
        smp = self.sub_matrix_precision
        assert tensor.shape == (smp.out_features, smp.in_features), (
            f"weight shape {tuple(tensor.shape)} does not match the precision grid for a "
            f"{smp.out_features}x{smp.in_features} matrix ({smp.name})"
        )
        tile_rows = [[None] * smp.num_col_tiles for _ in range(smp.num_row_tiles)]
        for r, c, row_sl, col_sl, _ in smp.iter_tiles():
            tile_rows[r][c] = self.quantizers[r][c](tensor[row_sl, col_sl])
        return torch.cat([torch.cat(row, dim=1) for row in tile_rows], dim=0)

    def set_scale_and_zp(self):
        for row in self.quantizers:
            for quantizer in row:
                quantizer.set_scale_and_zp()


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
        weight_bit_prec: WeightPrecSpec,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        bias: bool,
        noise_config: Optional[WeightNoiseConfig] = None,
        weight_dropout: float = 0.0,
        pruning_config: Optional[WeightPruningConfig] = None,
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
        sub_matrix_precision = SubMatrixPrecision.from_spec(
            weight_bit_prec, out_features, in_features, name=f"LinearQuant({out_features}x{in_features})"
        )
        if sub_matrix_precision is None:
            self.weight_quantizer = WeightQuantizer(
                bit_precision=self.weight_bit_prec,
                dtype=self.weight_quant_dtype,
                method=self.weight_quant_method,
            )
        else:
            self.weight_quantizer = TiledWeightQuantizer(
                sub_matrix_precision,
                dtype=self.weight_quant_dtype,
                method=self.weight_quant_method,
            )
        self.noise_config = noise_config
        self.weight_dropout = weight_dropout
        self.pruning_config = pruning_config

    def quantize_and_prune(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        """Fake-quantize and prune the weight, ordered by ``pruning_config.prune_before_quant``.

        With ``prune_before_quant=True`` the prune mask is computed on the raw (continuous) weight
        and the result is then quantized; otherwise the weight is quantized first and pruning is
        applied to the fake-quantized weight, so the zeros land on the quantization grid. If no
        pruning is configured this is just the weight quantizer.
        """
        if self.pruning_config is not None and self.pruning_config.prune_before_quant:
            weight = self.pruning_config.apply(weight, training)
            weight = self.weight_quantizer(weight)
        else:
            weight = self.weight_quantizer(weight)
            if self.pruning_config is not None:
                weight = self.pruning_config.apply(weight, training)
        return weight

    def prune_for_memristor_init(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        """Prune the weight that is handed to ``init_from_linear_quant`` (which quantizes it itself).

        With ``prune_before_quant=True`` the raw weight is pruned and the memristor init performs
        the quantization (identical to the un-pruned path). With ``prune_before_quant=False`` the
        weight is fake-quantized first and then pruned, so the zeros lie on the grid; the
        quantization that ``init_from_linear_quant`` re-applies with the same (frozen) scale is
        idempotent and preserves them. If no pruning is configured the weight is returned unchanged.
        """
        if self.pruning_config is None:
            return weight
        if self.pruning_config.prune_before_quant:
            return self.pruning_config.apply(weight, training)
        return self.pruning_config.apply(self.weight_quantizer(weight), training)

    def forward(self, tensor: torch.Tensor):
        weight = self.quantize_and_prune(self.weight, self.training)
        if self.weight_dropout > 0.0 and self.training:
            weight = F.dropout(weight, p=self.weight_dropout, training=True)
        if self.noise_config is not None:
            weight = self.noise_config.apply(weight, self.weight_quantizer, self.weight_bit_prec, self.training)
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
        noise_config: Optional[WeightNoiseConfig] = None,
        padding_mode: str = "zeros",
        weight_dropout: float = 0.0,
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

        assert isinstance(weight_bit_prec, (int, float)), (
            f"per-tile precision specs are not supported for conv weights, got {weight_bit_prec!r}"
        )
        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec,
            dtype=self.weight_quant_dtype,
            method=self.weight_quant_method,
        )
        self.noise_config = noise_config
        self.weight_dropout = weight_dropout

    def forward(self, tensor: torch.Tensor):
        weight = self.weight_quantizer(self.weight)
        if self.weight_dropout > 0.0 and self.training:
            weight = F.dropout(weight, p=self.weight_dropout, training=True)
        if self.noise_config is not None:
            weight = self.noise_config.apply(weight, self.weight_quantizer, self.weight_bit_prec, self.training)
        result = F.conv1d(
            tensor, weight, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return result


class Conv2dQuant(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        weight_bit_prec: int,
        weight_quant_dtype: torch.dtype,
        weight_quant_method: str,
        bias: bool,
        stride: Union[int, Tuple[int, int]],
        padding: Union[str, int, Tuple[int, int]],
        dilation: int,
        groups: int,
        padding_mode: str = "zeros",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        self.kernel_size = kernel_size
        if isinstance(stride, int):
            stride = (stride, stride)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels // groups, *kernel_size),
            requires_grad=True,
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.bias = None
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                init.uniform_(self.bias, -bound, bound)

        assert isinstance(weight_bit_prec, (int, float)), (
            f"per-tile precision specs are not supported for conv weights, got {weight_bit_prec!r}"
        )
        self.weight_bit_prec = weight_bit_prec
        self.weight_quant_dtype = weight_quant_dtype
        self.weight_quant_method = weight_quant_method
        self.weight_quantizer = WeightQuantizer(
            bit_precision=self.weight_bit_prec,
            dtype=self.weight_quant_dtype,
            method=self.weight_quant_method,
        )

    def forward(self, tensor: torch.Tensor):
        result = F.conv2d(
            tensor,
            self.weight_quantizer(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        return result

class _LinearQuantTileShim:
    """Duck-typed stand-in for a LinearQuant covering a single weight tile.

    Exposes exactly the attributes TiledMemristorLinear.init_from_linear_quant reads
    (.weight, .weight_quantizer incl. .scale, .bias).
    """

    def __init__(self, weight: torch.Tensor, weight_quantizer: WeightQuantizer):
        self.weight = weight
        self.weight_quantizer = weight_quantizer
        self.bias = None


class MixedPrecisionTiledMemristorLinear(nn.Module):
    """Memristor linear with per-tile weight precision, built from one TiledMemristorLinear per
    128x128 tile of the weight matrix.

    Numerically consistent with the uniform TiledMemristorLinear: that module also converts
    every tile separately through the ADC before summing, so applying each tile's own
    output_factor (which contains its own weight scale) before the sum is exact.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        sub_matrix_precision: SubMatrixPrecision,
        converter_hardware_settings,
        bias: bool = True,
    ):
        super().__init__()
        from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear

        smp = sub_matrix_precision
        assert smp.out_features == out_features and smp.in_features == in_features, (
            f"precision grid is for a {smp.out_features}x{smp.in_features} matrix, "
            f"module has {out_features}x{in_features} ({smp.name})"
        )
        self.in_features = in_features
        self.out_features = out_features
        self.sub_matrix_precision = smp
        tile_rows = [[None] * smp.num_col_tiles for _ in range(smp.num_row_tiles)]
        for r, c, row_sl, col_sl, prec in smp.iter_tiles():
            assert float(prec).is_integer() or prec == 1.5, (
                f"memristor conversion supports integer tile precisions and 1.5 only, "
                f"got {prec} for tile ({r}, {c}) ({smp.name})"
            )
            tile_rows[r][c] = TiledMemristorLinear(
                in_features=col_sl.stop - col_sl.start,
                out_features=row_sl.stop - row_sl.start,
                weight_precision=int(prec) if not prec == 1.5 else 2,
                converter_hardware_settings=converter_hardware_settings,
                memristor_inputs=smp.tile_size,
                memristor_outputs=smp.tile_size,
                bias=False,  # the bias lives on the wrapper, tiles must stay bias-free
            )
        self.tiles = nn.ModuleList(nn.ModuleList(row) for row in tile_rows)
        if bias:
            self.bias = nn.Parameter(torch.empty((out_features,)), requires_grad=True)
        else:
            self.bias = None

    @property
    def initialized(self) -> bool:
        return all(tile.initialized for row in self.tiles for tile in row)

    @initialized.setter
    def initialized(self, value: bool):
        for row in self.tiles:
            for tile in row:
                tile.initialized = value

    def init_from_linear_quant(
        self,
        activation_quant: ActivationQuantizer,
        linear_quant: LinearQuant,
        num_cycles_init: int,
        correction_settings,
    ):
        assert isinstance(linear_quant.weight_quantizer, TiledWeightQuantizer), (
            "MixedPrecisionTiledMemristorLinear must be initialized from a LinearQuant with a "
            "TiledWeightQuantizer (per-tile scales)"
        )
        assert linear_quant.weight_quantizer.sub_matrix_precision.grid == self.sub_matrix_precision.grid, (
            "precision grid of the LinearQuant does not match this module's grid"
        )
        for r, c, row_sl, col_sl, _prec in self.sub_matrix_precision.iter_tiles():
            tile_quantizer = linear_quant.weight_quantizer.get_tile_quantizer(r, c)
            shim = _LinearQuantTileShim(linear_quant.weight.data[row_sl, col_sl], tile_quantizer)
            self.tiles[r][c].init_from_linear_quant(
                activation_quant=activation_quant,
                linear_quant=shim,
                num_cycles_init=num_cycles_init,
                correction_settings=correction_settings,
            )
        self.bias = linear_quant.bias

    def forward(self, inputs: torch.Tensor):
        smp = self.sub_matrix_precision
        row_outputs = []
        for r in range(smp.num_row_tiles):
            acc = None
            for c in range(smp.num_col_tiles):
                col_sl = slice(c * smp.tile_size, min((c + 1) * smp.tile_size, smp.in_features))
                out = self.tiles[r][c](inputs[..., col_sl])
                acc = out if acc is None else acc + out
            row_outputs.append(acc)
        out = torch.cat(row_outputs, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out


def convert_linear_to_memristor(
    linear_quant: LinearQuant,
    activation_quant: ActivationQuantizer,
    converter_hardware_settings,
    num_cycles: int,
    correction_settings,
    out_features: Optional[int] = None,
) -> nn.Module:
    """Convert a trained LinearQuant into a memristor linear, shared by all prep_quant sites.

    Uniform precisions produce a plain TiledMemristorLinear (identical to the previous per-site
    code); per-tile specs produce a MixedPrecisionTiledMemristorLinear.

    :param out_features: optional override of the memristor out_features (only the output linear
        uses this to reserve twice its rows); unsupported for mixed precision layers.
    """
    from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear

    linear_quant.weight_quantizer.set_scale_and_zp()
    activation_quant.set_scale_and_zp()
    if linear_quant.pruning_config is not None:
        with torch.no_grad():
            linear_quant.weight.data = linear_quant.prune_for_memristor_init(
                linear_quant.weight.data, training=False
            )
    if isinstance(linear_quant.weight_quantizer, TiledWeightQuantizer):
        assert out_features is None, "out_features override is not supported for mixed precision layers"
        mem_lin = MixedPrecisionTiledMemristorLinear(
            in_features=linear_quant.in_features,
            out_features=linear_quant.out_features,
            sub_matrix_precision=linear_quant.weight_quantizer.sub_matrix_precision,
            converter_hardware_settings=converter_hardware_settings,
            bias=linear_quant.bias is not None,
        )
    else:
        mem_lin = TiledMemristorLinear(
            in_features=linear_quant.in_features,
            out_features=out_features if out_features is not None else linear_quant.out_features,
            weight_precision=linear_quant.weight_bit_prec if not linear_quant.weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
        )
    mem_lin.init_from_linear_quant(
        activation_quant=activation_quant,
        linear_quant=linear_quant,
        num_cycles_init=num_cycles,
        correction_settings=correction_settings,
    )
    return mem_lin


def make_inference_memristor_linear(
    in_features: int,
    out_features: int,
    weight_bit_prec: WeightPrecSpec,
    converter_hardware_settings,
    bias: bool = True,
    name: str = "",
) -> nn.Module:
    """Construct an (uninitialized) inference memristor linear for the mem_inited variants.

    The weights are loaded afterwards from a checkpoint produced by convert_linear_to_memristor,
    so the module tree built here must match the converted one exactly.
    """
    from synaptogen_ml.memristor_modules.linear import TiledMemristorLinear

    sub_matrix_precision = SubMatrixPrecision.from_spec(weight_bit_prec, out_features, in_features, name=name)
    if sub_matrix_precision is None:
        return TiledMemristorLinear(
            in_features=in_features,
            out_features=out_features,
            weight_precision=weight_bit_prec if not weight_bit_prec == 1.5 else 2,
            converter_hardware_settings=converter_hardware_settings,
            memristor_inputs=128,
            memristor_outputs=128,
            bias=bias,
        )
    return MixedPrecisionTiledMemristorLinear(
        in_features=in_features,
        out_features=out_features,
        sub_matrix_precision=sub_matrix_precision,
        converter_hardware_settings=converter_hardware_settings,
        bias=bias,
    )


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
        self.noise_config = cfg.weight_noise
        self.activation_quant_dtype = cfg.activation_quant_dtype
        self.activation_quant_method = cfg.activation_quant_method
        self.dot_quant_dtype = cfg.dot_quant_dtype
        self.dot_quant_method = cfg.dot_quant_method
        self.Av_quant_dtype = cfg.Av_quant_dtype
        self.Av_quant_method = cfg.Av_quant_method
        self.converter_hardware_settings = cfg.converter_hardware_settings
        self.pos_enc_converter_hardware_settings = cfg.pos_enc_converter_hardware_settings


        assert not self.learnable_pos_emb or self.rel_pos_clip

        self.att_weights_dropout = nn.Dropout(cfg.att_weights_dropout)

        assert self.embed_dim % self.num_heads == 0, "embed_dim must be divisible by num_heads"

        # projection matrices
        self.qkv_proj = LinearQuant(
            in_features=self.embed_dim,
            out_features=3 * self.embed_dim,
            weight_bit_prec=cfg.bit_prec_W_i,
            weight_quant_dtype=self.weight_quant_dtype,
            weight_quant_method=self.weight_quant_method,
            bias=cfg.with_bias,
            noise_config=self.noise_config,
            weight_dropout=cfg.weight_dropout,
            pruning_config=cfg.weight_pruning,
        )
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
        self.q_quantizer = ActivationQuantizer(
            cfg.activation_bit_prec,
            self.dot_quant_dtype,
            self.dot_quant_method,
            channel_axis=None if self.dot_quant_method == "per_tensor" else 3,
            moving_avrg=cfg.moving_average,
        )
        self.k_quantizer = ActivationQuantizer(
            cfg.activation_bit_prec,
            self.dot_quant_dtype,
            self.dot_quant_method,
            channel_axis=None if self.dot_quant_method == "per_tensor" else 2,
            moving_avrg=cfg.moving_average,
        )

        self.out_proj = LinearQuant(
            in_features=self.embed_dim,
            out_features=self.embed_dim,
            weight_bit_prec=cfg.bit_prec_W_o,
            weight_quant_dtype=self.weight_quant_dtype,
            weight_quant_method=self.weight_quant_method,
            bias=cfg.with_bias,
            noise_config=self.noise_config,
            weight_dropout=cfg.weight_dropout,
            pruning_config=cfg.weight_pruning,
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
                noise_config=self.noise_config,
                weight_dropout=cfg.weight_dropout,
                pruning_config=cfg.weight_pruning,
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

        q = query_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T, #heads, F']§
        k = key_seq.view(batch_dim_size, -1, self.num_heads, self.embed_dim_per_head)  # [B, T', #heads, F']
        q = self.q_quantizer(q)
        k = self.k_quantizer(k)


        if self.learnable_pos_emb:
            kv_pos_vec = torch.arange(time_dim_size, device=input_tensor.device)  # [kv_len]

            query_spatial_dim_m1 = time_dim_size - 1
            q_pos_vec = torch.arange(query_spatial_dim_m1, device=input_tensor.device)  # [q_len-1]

            # The min value is with kv_pos=0, q_pos=q_len-1: -(q_len-1)
            # The max value is with kv_pos=kv_len-1, q_pos=0: k_len-1
            indices = torch.concat((q_pos_vec - query_spatial_dim_m1, kv_pos_vec), dim=-1)
            indices = torch.clamp(indices, -self.rel_pos_clip, self.rel_pos_clip)
            # Shift values to be >= 0. Each integer still uniquely identifies a relative position difference.
            indices = indices + self.rel_pos_clip
            rel_pos_embeddings = self.rel_pos_embeddings[indices]  # [out_spatial_dim,n_out]
            rel_pos_embeddings = rel_pos_embeddings.unsqueeze(0)
            assert rel_pos_embeddings.shape == (1, 2 * time_dim_size - 1,
                                                self.pos_emb_dim), "Something went wrong in reshaping"
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

        attn_bd = self._rel_shift_bhij(attn_bd, k_len=time_dim_size)  # [B, #heads, T, T']
        assert attn_bd.shape == (batch_dim_size, self.num_heads, time_dim_size, time_dim_size)

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
        self.out_proj = convert_linear_to_memristor(
            linear_quant=self.out_proj,
            activation_quant=self.out_proj_in_quant,
            converter_hardware_settings=self.converter_hardware_settings,
            num_cycles=self.cfg.num_cycles,
            correction_settings=self.cfg.correction_settings,
        )
        self.out_proj_in_quant = nn.Identity()

        self.qkv_proj = convert_linear_to_memristor(
            linear_quant=self.qkv_proj,
            activation_quant=self.in_proj_in_quant,
            converter_hardware_settings=self.converter_hardware_settings,
            num_cycles=self.cfg.num_cycles,
            correction_settings=self.cfg.correction_settings,
        )
        self.in_proj_in_quant = nn.Identity()

        if self.cfg.with_linear_pos:
            self.linear_pos = convert_linear_to_memristor(
                linear_quant=self.linear_pos,
                activation_quant=self.learn_emb_in_quant,
                converter_hardware_settings=self.pos_enc_converter_hardware_settings
                if self.pos_enc_converter_hardware_settings is not None
                else self.converter_hardware_settings,
                num_cycles=self.cfg.num_cycles,
                correction_settings=self.cfg.correction_settings,
            )
            self.learn_emb_in_quant = nn.Identity()

        print("Finished MHSA")
