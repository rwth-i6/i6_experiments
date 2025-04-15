"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Union

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from torch_memristor.memristor_modules import DacAdcHardwareSettings


@dataclass(kw_only=True)
class VGG4LayerActFrontendV1Config_mod(VGG4LayerActFrontendV1Config):
    activation_str: str = ""
    activation: Optional[Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]] = None

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
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu


@dataclass
class QuantizedMultiheadAttentionV4Config(ModelConfiguration):

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    weight_quant_dtype: torch.dtype
    weight_quant_method: str
    activation_quant_dtype: torch.dtype
    activation_quant_method: str
    dot_quant_dtype: torch.dtype
    dot_quant_method: str
    Av_quant_dtype: torch.dtype
    Av_quant_method: str
    bit_prec_W_q: Union[int, float]
    bit_prec_W_k: Union[int, float]
    bit_prec_W_v: Union[int, float]
    bit_prec_dot: Union[int, float]
    bit_prec_A_v: Union[int, float]
    bit_prec_W_o: Union[int, float]
    activation_bit_prec: Union[int, float]
    moving_average: Optional[float]  # Moving average for input quantization
    dropout: float
    quant_in_linear: bool
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


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

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


@dataclass
class ConformerBlockQuantV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for ConformerMHSAV1
        conv_cfg: Configuration for ConformerConvolutionV1
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardQuantV4Config
    mhsa_cfg: QuantizedMultiheadAttentionV4Config
    conv_cfg: ConformerConvolutionQuantV4Config


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
    block_cfg: ConformerBlockQuantV1Config


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
class QuantModelTrainConfigV4:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
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
    weight_quant_dtype: Union[torch.dtype, str]
    weight_quant_method: str
    activation_quant_dtype: Union[torch.dtype, str]
    activation_quant_method: str
    dot_quant_dtype: Union[torch.dtype, str]
    dot_quant_method: str
    Av_quant_dtype: Union[torch.dtype, str]
    Av_quant_method: str
    moving_average: Optional[float]  # default if enabled should be 0.01, if set enables moving average
    weight_bit_prec: Union[int, float]
    activation_bit_prec: Union[int, float]
    quantize_output: bool
    quant_in_linear: bool
    converter_hardware_settings: Optional[DacAdcHardwareSettings]
    num_cycles: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["converter_hardware_settings"] = DacAdcHardwareSettings(**d["converter_hardware_settings"])
        for name in ["weight_quant_dtype", "activation_quant_dtype", "dot_quant_dtype", "Av_quant_dtype"]:
            if d[name] == "qint8":
                weight_dtype = torch.qint8
            elif d[name] == "quint8":
                weight_dtype = torch.quint8
            else:
                raise NotImplementedError
            d[name] = weight_dtype
        return QuantModelTrainConfigV4(**d)

    def __post_init__(self):
        for param in [self.weight_quant_dtype, self.activation_quant_dtype, self.dot_quant_dtype, self.Av_quant_dtype]:
            if param == "qint8":
                param = torch.qint8
            elif param == "quint8":
                param = torch.quint8
            elif any(param == x for x in [torch.quint8, torch.qint8]):
                continue
            else:
                raise NotImplementedError
