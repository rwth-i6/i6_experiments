"""
v14 adds weight sparsity / pruning
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from typing import Callable, Optional, Union, Dict, Literal, List, Tuple

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from synaptogen_ml.memristor_modules.memristor import DacAdcHardwareSettings
from synaptogen_ml.memristor_modules.config import CycleCorrectionSettings


@dataclass
class ThresholdPruningConfig:
    """Prune weights whose absolute value is below a fixed threshold.

    :param prune_before_quant: if True the prune mask is computed and applied on the raw
        (continuous) weight *before* the weight quantizer; if False pruning is applied to the
        already fake-quantized weight, so the zeros live on the quantization grid.
    """
    start_epoch: int
    threshold: float
    prune_before_quant: bool

    def apply(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        return weight * (weight.abs() >= self.threshold).to(weight.dtype)


@dataclass
class PercentilePruningConfig:
    """Prune the bottom `percentile` fraction of weights by absolute value (value in [0, 1]).

    :param prune_before_quant: if True the prune mask is computed and applied on the raw
        (continuous) weight *before* the weight quantizer; if False pruning is applied to the
        already fake-quantized weight, so the zeros live on the quantization grid. Computing the
        percentile on the continuous weights gives the requested sparsity more reliably, since the
        fake-quantized weights are tied to a discrete grid.
    """
    start_epoch: int
    percentile: float
    prune_before_quant: bool

    def apply(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        cutoff = torch.quantile(weight.abs(), self.percentile)
        return weight * (weight.abs() >= cutoff).to(weight.dtype)


WeightPruningConfig = Union[ThresholdPruningConfig, PercentilePruningConfig]


# A weight precision spec for a single matrix: a scalar (uniform precision, legacy behavior),
# a 1D list (one precision per output tile row, broadcast across input tiles) or a 2D nested
# list (one precision per tile). Lists are JSON-serializable, so specs pass through
# dataclasses.asdict / from_dict unchanged.
WeightPrecSpec = Union[int, float, List]

TILE_SIZE = 128  # memristor subarray edge length, fixed by the hardware
_ALLOWED_PRECISIONS = (1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8)


class SubMatrixPrecision:
    """Normalized per-tile precision grid for one weight matrix W [out_features, in_features].

    The matrix is split into ``tile_size x tile_size`` submatrices (matching the memristor
    subarrays); ``grid[r][c]`` is the bit precision of the tile
    ``W[r*tile_size:(r+1)*tile_size, c*tile_size:(c+1)*tile_size]``. Edge tiles are clipped to
    the matrix bounds.
    """

    def __init__(
        self,
        grid: List[List[Union[int, float]]],
        out_features: int,
        in_features: int,
        name: str = "",
        tile_size: int = TILE_SIZE,
    ):
        self.grid = grid
        self.out_features = out_features
        self.in_features = in_features
        self.name = name
        self.tile_size = tile_size
        self.check()

    @classmethod
    def from_spec(
        cls,
        spec: WeightPrecSpec,
        out_features: int,
        in_features: int,
        name: str = "",
        tile_size: int = TILE_SIZE,
    ) -> Optional["SubMatrixPrecision"]:
        """Normalize a weight precision spec for a matrix of the given shape.

        Returns None for scalar specs, which keep the legacy uniform per-tensor quantization.
        """
        if isinstance(spec, (int, float)):
            return None
        assert isinstance(spec, (list, tuple)) and len(spec) > 0, (
            f"{name}: invalid weight precision spec {spec!r}"
        )
        if all(isinstance(p, (int, float)) for p in spec):
            # 1D shorthand: one precision per output tile row, broadcast across input tiles
            num_col_tiles = (in_features + tile_size - 1) // tile_size
            grid = [[p] * num_col_tiles for p in spec]
        else:
            grid = [list(row) if isinstance(row, (list, tuple)) else row for row in spec]
        return cls(grid=grid, out_features=out_features, in_features=in_features, name=name, tile_size=tile_size)

    def check(self):
        num_row_tiles = (self.out_features + self.tile_size - 1) // self.tile_size
        num_col_tiles = (self.in_features + self.tile_size - 1) // self.tile_size
        assert len(self.grid) == num_row_tiles, (
            f"{self.name}: precision grid has {len(self.grid)} tile rows, but a "
            f"{self.out_features}x{self.in_features} matrix with tile size {self.tile_size} "
            f"needs {num_row_tiles}"
        )
        for r, row in enumerate(self.grid):
            assert isinstance(row, (list, tuple)) and len(row) == num_col_tiles, (
                f"{self.name}: precision grid row {r} is {row!r}, but a "
                f"{self.out_features}x{self.in_features} matrix with tile size {self.tile_size} "
                f"needs {num_col_tiles} entries per row"
            )
            for c, prec in enumerate(row):
                assert isinstance(prec, (int, float)) and prec in _ALLOWED_PRECISIONS, (
                    f"{self.name}: invalid precision {prec!r} for tile ({r}, {c}), "
                    f"allowed are {_ALLOWED_PRECISIONS}"
                )

    @property
    def num_row_tiles(self) -> int:
        return len(self.grid)

    @property
    def num_col_tiles(self) -> int:
        return len(self.grid[0])

    @property
    def max_precision(self) -> Union[int, float]:
        return max(max(row) for row in self.grid)

    def iter_tiles(self):
        """Yields (r, c, row_slice, col_slice, precision) for every tile, edge tiles clipped."""
        for r in range(self.num_row_tiles):
            row_sl = slice(r * self.tile_size, min((r + 1) * self.tile_size, self.out_features))
            for c in range(self.num_col_tiles):
                col_sl = slice(c * self.tile_size, min((c + 1) * self.tile_size, self.in_features))
                yield r, c, row_sl, col_sl, self.grid[r][c]

    def __repr__(self):
        return (
            f"SubMatrixPrecision(name={self.name!r}, shape={self.out_features}x{self.in_features}, "
            f"tile_size={self.tile_size}, grid={self.grid})"
        )


def split_ff_spec(spec: Union[WeightPrecSpec, Dict[str, WeightPrecSpec]]) -> Tuple[WeightPrecSpec, WeightPrecSpec]:
    """Split a feed-forward module precision spec into (lin_1, lin_2).

    A scalar applies to both linears; non-scalar specs must be given per matrix as
    ``{"lin_1": spec, "lin_2": spec}`` since the two linears have transposed shapes.
    """
    if isinstance(spec, dict):
        assert set(spec.keys()) == {"lin_1", "lin_2"}, (
            f"ff precision dict must have exactly the keys {{'lin_1', 'lin_2'}}, got {set(spec.keys())}"
        )
        return spec["lin_1"], spec["lin_2"]
    assert isinstance(spec, (int, float)), (
        f"non-scalar ff precision specs must be given per matrix as a dict "
        f"{{'lin_1': ..., 'lin_2': ...}} (the linears have different shapes), got {spec!r}"
    )
    return spec, spec


def split_mhsa_spec(
    spec: Union[WeightPrecSpec, Dict[str, WeightPrecSpec]], with_linear_pos: bool
) -> Tuple[WeightPrecSpec, WeightPrecSpec, WeightPrecSpec]:
    """Split an MHSA module precision spec into (W_i, W_o, learn_emb).

    A scalar applies to all projections; non-scalar specs must be given per matrix as
    ``{"W_i": spec, "W_o": spec}`` plus ``"learn_emb"`` iff ``with_linear_pos`` is set.
    """
    if isinstance(spec, dict):
        expected = {"W_i", "W_o"} | ({"learn_emb"} if with_linear_pos else set())
        assert set(spec.keys()) == expected, (
            f"mhsa precision dict must have exactly the keys {expected} "
            f"(with_linear_pos={with_linear_pos}), got {set(spec.keys())}"
        )
        return spec["W_i"], spec["W_o"], spec.get("learn_emb", spec["W_i"])
    assert isinstance(spec, (int, float)), (
        f"non-scalar mhsa precision specs must be given per matrix as a dict "
        f"{{'W_i': ..., 'W_o': ...[, 'learn_emb': ...]}} (the projections have different shapes), got {spec!r}"
    )
    return spec, spec, spec


def split_conv_spec(
    spec: Union[WeightPrecSpec, Dict[str, WeightPrecSpec]]
) -> Tuple[WeightPrecSpec, WeightPrecSpec, Union[int, float]]:
    """Split a conv module precision spec into (pconv_1, pconv_2, dconv).

    A scalar applies to all three; non-scalar specs must be given per matrix as
    ``{"pconv_1": spec, "pconv_2": spec, "dconv": int}``. The depthwise conv weight is not a
    matrix mapped onto 128x128 subarrays, so its precision must stay scalar.
    """
    if isinstance(spec, dict):
        assert set(spec.keys()) == {"pconv_1", "pconv_2", "dconv"}, (
            f"conv precision dict must have exactly the keys {{'pconv_1', 'pconv_2', 'dconv'}}, "
            f"got {set(spec.keys())}"
        )
        assert isinstance(spec["dconv"], (int, float)), (
            f"depthwise conv precision must be a scalar, got {spec['dconv']!r}"
        )
        return spec["pconv_1"], spec["pconv_2"], spec["dconv"]
    assert isinstance(spec, (int, float)), (
        f"non-scalar conv precision specs must be given per matrix as a dict "
        f"{{'pconv_1': ..., 'pconv_2': ..., 'dconv': ...}} (the matrices have different shapes), got {spec!r}"
    )
    return spec, spec, spec


def _apply_noise_per_tile(noise_config, weight: torch.Tensor, weight_quantizer) -> torch.Tensor:
    """Apply a weight noise config tile-by-tile for a TiledWeightQuantizer (duck-typed via
    ``is_tiled`` to avoid a circular import of the modules file)."""
    out = weight.clone()
    for r, c, row_sl, col_sl, prec in weight_quantizer.sub_matrix_precision.iter_tiles():
        assert float(prec).is_integer(), (
            f"weight noise requires integer tile precisions, got {prec} for tile ({r}, {c}) "
            f"of {weight_quantizer.sub_matrix_precision.name}"
        )
        tile_quantizer = weight_quantizer.get_tile_quantizer(r, c)
        out[row_sl, col_sl] = noise_config._apply_single(weight[row_sl, col_sl], tile_quantizer, int(prec))
    return out


@dataclass
class GaussianWeightNoiseConfig:
    """Per-bit Gaussian noise simulating memristor read uncertainty."""
    dev: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        if getattr(weight_quantizer, "is_tiled", False):
            return _apply_noise_per_tile(self, weight, weight_quantizer)
        return self._apply_single(weight, weight_quantizer, weight_bit_prec)

    def _apply_single(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int) -> torch.Tensor:
        for i in range(weight_bit_prec - 1):
            mean = 2 * (-weight_quantizer.zero_point).expand(weight.shape).to(weight.device).to(torch.float32)
            std = (torch.tensor(self.dev) * (2 ** i)).expand(weight.shape).to(weight.device).to(torch.float32)
            std = 2**0.5 * std
            noise = torch.normal(mean=mean, std=std).to(weight.device) * weight_quantizer.scale
            weight = weight + noise
        return weight


@dataclass
class BitFlipWeightNoiseConfig:
    """Per-bit random bit-flip noise simulating memristor programming errors.

    Each bit of the quantized weight is independently flipped (0→1 or 1→0)
    with probability `p`, modelling a device being programmed into the wrong state.
    """
    p: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        if getattr(weight_quantizer, "is_tiled", False):
            return _apply_noise_per_tile(self, weight, weight_quantizer)
        return self._apply_single(weight, weight_quantizer, weight_bit_prec)

    def _apply_single(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int) -> torch.Tensor:
        scale = weight_quantizer.scale.to(weight.device).to(torch.float32)
        zero_point = weight_quantizer.zero_point.to(weight.device).to(torch.int32)
        n = weight_bit_prec
        bit_mask = (1 << n) - 1
        # Re-quantize to signed integer, then extract the n-bit pattern
        q_int = (weight.to(torch.float32) / scale).round().to(torch.int32) + zero_point
        q_bits = q_int & bit_mask
        # Flip each bit independently with probability p
        for i in range(n):
            flip_mask = torch.bernoulli(
                torch.full(weight.shape, self.p, dtype=torch.float32, device=weight.device)
            ).to(torch.int32)
            q_bits = q_bits ^ (flip_mask * (1 << i))
        q_bits = q_bits & bit_mask
        # Convert n-bit unsigned back to signed (two's complement)
        sign_bit_val = 1 << (n - 1)
        q_signed = (q_bits ^ sign_bit_val) - sign_bit_val
        # Dequantize
        return (q_signed - zero_point).to(weight.dtype) * scale


WeightNoiseConfig = Union[GaussianWeightNoiseConfig, BitFlipWeightNoiseConfig]


@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config_mod(VGG4LayerActFrontendV1Config):
    activation: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]]
    activation_str: str = ""

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        activation_str = d.pop("activation_str")
        if activation_str == "ReLU":
            from torch.nn import ReLU

            activation = ReLU()
        else:
            assert False, "Unsupported activation %s" % d["activation_str"]
        d["activation"] = activation
        return VGG4LayerActFrontendV1Config(**d)


@dataclass
class ConformerPositionwiseFeedForwardQuantV4Config(ModelConfiguration):
    """
    Attributes:
        input_dim: input dimension
        hidden_dim: hidden dimension (normally set to 4*input_dim as suggested by the paper)
        dropout: dropout probability
        activation: activation function
    """

    input_dim: int
    hidden_dim: int
    dropout: float
    weight_bit_prec: Union[int, float, Dict[str, WeightPrecSpec]]  # dict keys: "lin_1", "lin_2"
    activation_bit_prec: Union[int, float]
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    moving_average: Optional[float]  # Moving average for input quantization
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    weight_noise: Optional[WeightNoiseConfig]
    correction_settings: Optional[CycleCorrectionSettings]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu

@dataclass
class ConformerPosEmbConfig(ModelConfiguration):
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float

@dataclass
class QuantizedConformerMHSARelPosV1Config(ModelConfiguration):

    input_dim: int
    num_att_heads: int
    with_bias: bool
    att_weights_dropout: float
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    dot_quant_dtype: torch.dtype
    dot_quant_method: str
    Av_quant_dtype: torch.dtype
    Av_quant_method: str
    bit_prec_W_i: WeightPrecSpec
    bit_prec_W_o: WeightPrecSpec
    bit_prec_learn_emb: WeightPrecSpec
    activation_bit_prec: Union[int, float]
    moving_average: Optional[float]  # Moving average for input quantization
    dropout: float
    quant_in_linear: bool
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    correction_settings: Optional[CycleCorrectionSettings]
    weight_noise: Optional[WeightNoiseConfig]
    learnable_pos_emb: bool
    rel_pos_clip: Optional[int]
    with_linear_pos: bool
    with_pos_bias: bool
    separate_pos_emb_per_head: bool
    pos_emb_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"
        assert self.dropout_broadcast_axes in [
            None,
            "B",
            "T",
            "BT",
        ], "invalid value, supported are None, 'B', 'T' and 'BT'"


@dataclass
class ConformerConvolutionQuantV4Config(ModelConfiguration):
    """
    Attributes:
        channels: number of channels for conv layers
        kernel_size: kernel size of conv layers
        dropout: dropout probability
        activation: activation function applied after normalization
        norm: normalization layer with input of shape [N,C,T]
    """

    channels: int
    kernel_size: int
    dropout: float
    activation: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    norm: Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]
    weight_bit_prec: Union[int, float, Dict[str, WeightPrecSpec]]  # dict keys: "pconv_1", "pconv_2", "dconv"
    activation_bit_prec: Union[int, float]
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    moving_average: Optional[float]  # Moving average for input quantization
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    correction_settings: Optional[CycleCorrectionSettings]
    weight_noise: Optional[WeightNoiseConfig]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


@dataclass
class ConformerBlockQuantV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1 (first ff, or both if ff2_cfg is None)
        ff2_cfg: Optional separate config for the second ff module; if None, ff_cfg is reused
        mhsa_cfg: Configuration for ConformerMHSAV1
        conv_cfg: Configuration for ConformerConvolutionV1
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardQuantV4Config
    mhsa_cfg: QuantizedConformerMHSARelPosV1Config
    conv_cfg: ConformerConvolutionQuantV4Config
    ff2_cfg: Optional[ConformerPositionwiseFeedForwardQuantV4Config] = None
    modules: List[str] = field(default_factory=lambda: ["ff", "mhsa", "conv", "ff"])
    scales: List[float] = field(default_factory=lambda: [0.5, 1.0, 1.0, 0.5])


@dataclass
class ConformerEncoderQuantV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: Union[ConformerBlockQuantV1Config, List[ConformerBlockQuantV1Config]]


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return SpecaugConfig(**d)


@dataclass
class QuantModelTrainConfigV16:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    pos_emb_config: ConformerPosEmbConfig
    specauc_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    conv_kernel_size: int
    final_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    weight_quant_dtype: Union[torch.dtype, str]
    weight_quant_method: str
    activation_quant_dtype: Union[torch.dtype, str]
    activation_quant_method: str
    dot_quant_dtype: Union[torch.dtype, str]
    dot_quant_method: str
    Av_quant_dtype: Union[torch.dtype, str]
    Av_quant_method: str
    moving_average: Optional[float]  # default if enabled should be 0.01, if set enables moving average
    # scalar, or one entry per layer; a layer entry is a scalar or a dict with keys
    # ff/ff1/ff2/mhsa/conv whose values are scalars or per-matrix dicts of WeightPrecSpec
    # (see split_ff_spec / split_mhsa_spec / split_conv_spec and SubMatrixPrecision)
    weight_bit_prec: Union[int, float, List[Union[int, float, Dict[str, Union[int, float, List, Dict[str, WeightPrecSpec]]]]]]
    activation_bit_prec: Union[int, float]
    quantize_output: bool
    quant_in_linear: bool
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    pos_enc_converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int
    correction_settings: Optional[CycleCorrectionSettings]
    weight_noise: Optional[WeightNoiseConfig]
    module_list: List[str]
    module_scales: List[float]
    aux_ctc_loss_layers: Optional[List[int]]
    aux_ctc_loss_scales: Optional[List[float]]
    weight_dropout: float
    weight_pruning: Optional[WeightPruningConfig]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["converter_hardware_settings"] = DacAdcHardwareSettings(**d["converter_hardware_settings"]) if d["converter_hardware_settings"] is not None else None
        d["pos_enc_converter_hardware_settings"] = DacAdcHardwareSettings(**d["pos_enc_converter_hardware_settings"]) if d["pos_enc_converter_hardware_settings"] is not None else None
        d["correction_settings"] = CycleCorrectionSettings(**d["correction_settings"]) if d["correction_settings"] is not None else None
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        if d.get("weight_pruning") is not None:
            pruning_d = d["weight_pruning"]
            if "threshold" in pruning_d:
                d["weight_pruning"] = ThresholdPruningConfig(
                    start_epoch=pruning_d["start_epoch"],
                    threshold=pruning_d["threshold"],
                    prune_before_quant=pruning_d["prune_before_quant"],
                )
            elif "percentile" in pruning_d:
                d["weight_pruning"] = PercentilePruningConfig(
                    start_epoch=pruning_d["start_epoch"],
                    percentile=pruning_d["percentile"],
                    prune_before_quant=pruning_d["prune_before_quant"],
                )
            else:
                raise NotImplementedError(f"Cannot determine pruning type from keys: {list(pruning_d.keys())}")
        else:
            d["weight_pruning"] = None

        for name in ["weight_quant_dtype", "activation_quant_dtype", "dot_quant_dtype", "Av_quant_dtype"]:
            if d[name] == "qint8":
                weight_dtype = torch.qint8
            elif d[name] == "quint8":
                weight_dtype = torch.quint8
            else:
                raise NotImplementedError
            d[name] = weight_dtype
        if d.get("weight_noise") is not None:
            noise_d = d["weight_noise"]
            if "dev" in noise_d:
                d["weight_noise"] = GaussianWeightNoiseConfig(dev=noise_d["dev"], start_epoch=noise_d["start_epoch"])
            elif "p" in noise_d:
                d["weight_noise"] = BitFlipWeightNoiseConfig(p=noise_d["p"], start_epoch=noise_d["start_epoch"])
            else:
                raise NotImplementedError(f"Cannot determine noise type from keys: {list(noise_d.keys())}")
        else:
            d["weight_noise"] = None
        return QuantModelTrainConfigV16(**d)

    def _validate_layer_weight_prec(self, entry, layer_idx: int):
        """Validate one per-layer weight_bit_prec entry, including per-matrix grid specs.

        Grid dimensions are checked against the actual matrix shapes here already, so invalid
        configs fail at experiment-definition (Sisyphus) time instead of on the cluster.
        """
        if isinstance(entry, (int, float)):
            return
        assert isinstance(entry, dict), (
            f"weight_bit_prec entry for layer {layer_idx} must be a scalar or a dict, got {entry!r} "
            f"(per-tile grids must be nested inside the per-matrix dicts)"
        )
        ff1_spec = entry.get("ff1", entry.get("ff"))
        ff2_spec = entry.get("ff2", entry.get("ff"))
        mhsa_spec = entry["mhsa"]
        conv_spec = entry["conv"]
        conf = self.conformer_size
        with_linear_pos = self.pos_emb_config.with_linear_pos
        separate = self.pos_emb_config.separate_pos_emb_per_head
        pos_emb_dim = conf if (with_linear_pos or separate) else conf // self.num_heads
        learn_emb_out = conf if separate else conf // self.num_heads
        for ff_name, ff_spec in [("ff1", ff1_spec), ("ff2", ff2_spec)]:
            lin1, lin2 = split_ff_spec(ff_spec)
            SubMatrixPrecision.from_spec(lin1, self.ff_dim, conf, name=f"layer {layer_idx} {ff_name}.lin_1")
            SubMatrixPrecision.from_spec(lin2, conf, self.ff_dim, name=f"layer {layer_idx} {ff_name}.lin_2")
        w_i, w_o, learn_emb = split_mhsa_spec(mhsa_spec, with_linear_pos=with_linear_pos)
        SubMatrixPrecision.from_spec(w_i, 3 * conf, conf, name=f"layer {layer_idx} mhsa.W_i")
        SubMatrixPrecision.from_spec(w_o, conf, conf, name=f"layer {layer_idx} mhsa.W_o")
        if with_linear_pos:
            SubMatrixPrecision.from_spec(
                learn_emb, learn_emb_out, pos_emb_dim, name=f"layer {layer_idx} mhsa.learn_emb"
            )
        pconv1, pconv2, _dconv = split_conv_spec(conv_spec)
        SubMatrixPrecision.from_spec(pconv1, 2 * conf, conf, name=f"layer {layer_idx} conv.pconv_1")
        SubMatrixPrecision.from_spec(pconv2, conf, conf, name=f"layer {layer_idx} conv.pconv_2")

    def __post_init__(self):
        if isinstance(self.weight_bit_prec, list):
            assert len(self.weight_bit_prec) == self.num_layers, (
                f"weight_bit_prec list length {len(self.weight_bit_prec)} must match num_layers {self.num_layers}"
            )
            assert self.quantize_output is not True, (
                "quantize_output=True requires a scalar weight_bit_prec "
                "(per-layer/per-tile specs do not define the output linear precision)"
            )
            _valid_keys = {"ff", "ff1", "ff2", "mhsa", "conv"}
            for layer_idx, entry in enumerate(self.weight_bit_prec):
                if isinstance(entry, dict):
                    assert set(entry.keys()) <= _valid_keys, (
                        f"weight_bit_prec dict keys must be a subset of {_valid_keys}, got {set(entry.keys())}"
                    )
                    assert "ff" in entry or ("ff1" in entry and "ff2" in entry), (
                        "weight_bit_prec dict must contain 'ff' or both 'ff1' and 'ff2'"
                    )
                    assert "mhsa" in entry, "weight_bit_prec dict must contain 'mhsa'"
                    assert "conv" in entry, "weight_bit_prec dict must contain 'conv'"
                self._validate_layer_weight_prec(entry, layer_idx)
        for param in [self.weight_quant_dtype, self.activation_quant_dtype, self.dot_quant_dtype, self.Av_quant_dtype]:
            if param == "qint8":
                param = torch.qint8
            elif param == "quint8":
                param = torch.quint8
            elif any(param == x for x in [torch.quint8, torch.qint8]):
                continue
            else:
                raise NotImplementedError
