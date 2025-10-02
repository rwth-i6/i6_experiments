"""
Config for the base CTC models without QAT v1, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Union

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ...trainers.train_handler import TrainMode


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
class ConformerPositionwiseFeedForwardNoQuantV1Config(ModelConfiguration):
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
    activation: Callable[[torch.Tensor], torch.Tensor] = nn.functional.silu


@dataclass
class MultiheadAttentionNoQuantV1Config(ModelConfiguration):

    input_dim: int
    num_att_heads: int
    att_weights_dropout: float
    dropout: float

    def __post_init__(self) -> None:
        super().__post_init__()
        assert self.input_dim % self.num_att_heads == 0, "input_dim must be divisible by num_att_heads"


@dataclass
class ConformerConvolutionNoQuantV1Config(ModelConfiguration):
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

    def check_valid(self):
        assert self.kernel_size % 2 == 1, "ConformerConvolutionNoQuantV1 only supports odd kernel sizes"

    def __post_init__(self):
        super().__post_init__()
        self.check_valid()


@dataclass
class ConformerBlockNoQuantV1Config(ModelConfiguration):
    """
    Attributes:
        ff_cfg: Configuration for ConformerPositionwiseFeedForwardV1
        mhsa_cfg: Configuration for ConformerMHSAV1
        conv_cfg: Configuration for ConformerConvolutionV1
    """

    # nested configurations
    ff_cfg: ConformerPositionwiseFeedForwardNoQuantV1Config
    mhsa_cfg: MultiheadAttentionNoQuantV1Config
    conv_cfg: ConformerConvolutionNoQuantV1Config


@dataclass
class ConformerEncoderNoQuantV1Config(ModelConfiguration):
    """
    Attributes:
        num_layers: Number of conformer layers in the conformer encoder
        frontend: A pair of ConformerFrontend and corresponding config
        block_cfg: Configuration for ConformerBlockV1
    """

    num_layers: int

    # nested configurations
    frontend: ModuleFactoryV1
    block_cfg: ConformerBlockNoQuantV1Config


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
class StreamableFeatureExtractorV1Config(ModelConfiguration):
    logmel_cfg: LogMelFeatureExtractionV1Config
    specaug_cfg: SpecaugConfig
    specaug_start_epoch: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["logmel_cfg"] = LogMelFeatureExtractionV1Config(**d["logmel_cfg"])
        d["specaug_cfg"] = SpecaugConfig(**d["specaug_cfg"])

        return StreamableFeatureExtractorV1Config(**d)


@dataclass
class ModelTrainNoQuantConfigV1:
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

    # streaming params
    chunk_size: Optional[float]  # in #samples
    lookahead_size: Optional[int]  # in #frames after frontend subsampling
    carry_over_size: Optional[int]  # in #chunks after frontend subsampling

    dual_mode: Optional[bool]  # separate linear or layernorm weights for offline and streaming mode (not implemented yet)
    streaming_scale: Optional[float]  # weight of streaming loss during training, only relevant for unified training

    train_mode: Union[str, TrainMode]  # offline, streaming or switching training


    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["train_mode"] = {str(strat): strat for strat in TrainMode}[d["train_mode"]]
        return ModelTrainNoQuantConfigV1(**d)