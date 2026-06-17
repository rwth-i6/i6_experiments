"""
v14 adds weight sparsity / pruning
"""

from dataclasses import dataclass, field

import torch
from torch import nn
from typing import Callable, Optional, Union, Dict, Literal, List

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from torch_memristor.memristor_modules import DacAdcHardwareSettings, CycleCorrectionSettings


@dataclass
class ThresholdPruningConfig:
    """Prune weights whose absolute value is below a fixed threshold."""
    start_epoch: int
    threshold: float

    def apply(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        return weight * (weight.abs() >= self.threshold).to(weight.dtype)


@dataclass
class PercentilePruningConfig:
    """Prune the bottom `percentile` fraction of weights by absolute value (value in [0, 1])."""
    start_epoch: int
    percentile: float

    def apply(self, weight: torch.Tensor, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
        cutoff = torch.quantile(weight.abs(), self.percentile)
        return weight * (weight.abs() >= cutoff).to(weight.dtype)


WeightPruningConfig = Union[ThresholdPruningConfig, PercentilePruningConfig]


@dataclass
class GaussianWeightNoiseConfig:
    """Per-bit Gaussian noise simulating memristor read uncertainty."""
    dev: float
    start_epoch: int

    def apply(self, weight: torch.Tensor, weight_quantizer, weight_bit_prec: int, training: bool) -> torch.Tensor:
        from returnn.torch.context import get_run_ctx
        if training and get_run_ctx().epoch < self.start_epoch:
            return weight
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
    weight_bit_prec: Union[int, float]
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
    bit_prec_W_i: Union[int, float]
    bit_prec_W_o: Union[int, float]
    bit_prec_learn_emb: Union[int, float]
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
    weight_bit_prec: Union[int, float]
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
class QuantModelTrainConfigV15:
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
    weight_bit_prec: Union[int, float, List[Union[int, float, Dict[str, Union[int, float]]]]]
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
                )
            elif "percentile" in pruning_d:
                d["weight_pruning"] = PercentilePruningConfig(
                    start_epoch=pruning_d["start_epoch"],
                    percentile=pruning_d["percentile"],
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
        return QuantModelTrainConfigV15(**d)

    def __post_init__(self):
        if isinstance(self.weight_bit_prec, list):
            assert len(self.weight_bit_prec) == self.num_layers, (
                f"weight_bit_prec list length {len(self.weight_bit_prec)} must match num_layers {self.num_layers}"
            )
            _valid_keys = {"ff", "ff1", "ff2", "mhsa", "conv"}
            for entry in self.weight_bit_prec:
                if isinstance(entry, dict):
                    assert set(entry.keys()) <= _valid_keys, (
                        f"weight_bit_prec dict keys must be a subset of {_valid_keys}, got {set(entry.keys())}"
                    )
                    assert "ff" in entry or ("ff1" in entry and "ff2" in entry), (
                        "weight_bit_prec dict must contain 'ff' or both 'ff1' and 'ff2'"
                    )
                    assert "mhsa" in entry, "weight_bit_prec dict must contain 'mhsa'"
                    assert "conv" in entry, "weight_bit_prec dict must contain 'conv'"
        for param in [self.weight_quant_dtype, self.activation_quant_dtype, self.dot_quant_dtype, self.Av_quant_dtype]:
            if param == "qint8":
                param = torch.qint8
            elif param == "quint8":
                param = torch.quint8
            elif any(param == x for x in [torch.quint8, torch.qint8]):
                continue
            else:
                raise NotImplementedError
