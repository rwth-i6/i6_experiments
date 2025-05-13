"""
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, List, Literal, Optional, Tuple, Union

from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from .i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
    VGG4LayerActFrontendV1Config_mod,
    SpecaugConfig,
    ConformerPosEmbConfig,
)
from ..features.scf import (
    SupervisedConvolutionalFeatureExtractionV1Config,
    SupervisedConvolutionalFeatureExtractionV2Config,
)
from ..features.stft import StftFeatureExtractionV1Config
from ..features.conv import ConvFeatureExtractionV1Config, ConvFeatureExtractionV2Config


@dataclass
class IdentityConfig(ModelConfiguration):
    module_class: str = "Identity"


@dataclass
class SpecaugStftConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int
    window_size: int
    window_shift: int
    fft_size: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return SpecaugStftConfig(**d)


@dataclass
class SpecaugMultiplierLinearConfig(ModelConfiguration):
    start_epoch: int
    end_epoch: int
    start_factor: float
    end_factor: float


@dataclass
class SpecaugStftV2Config(SpecaugStftConfig):
    min_num_time: int
    min_num_feat: int
    multiplier: Optional[SpecaugMultiplierLinearConfig] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        if d["multiplier"] is not None:
            d["multiplier"] = SpecaugMultiplierLinearConfig(**d["multiplier"])
        return SpecaugStftV2Config(**d)


@dataclass
class VGGNLayerActFrontendV1Config(ModelConfiguration):
    """
    Attributes:
        in_features: number of input features to module
        convs: description of N conv layers in list of length N that contains [(dim, kernel_size, stride), ...]
        activations: activations after each conv layer in list of length N (None for no activation)
        poolings: description of pooling after each conv layer in list of length N that contains
            [(kernel_size, stride, padding), ...] or None for no pooling
        out_features: output size of the final linear layer
    """

    in_features: int
    convs: List[Tuple[int, Tuple[int, int], Union[int, Tuple[int, int]]]]
    activations: List[Optional[str]]
    poolings: List[Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]]]
    out_features: int

    def check_valid(self):
        assert len(self.convs) == len(self.activations), \
            f"convs and activations must have same length: {self.convs} vs. {self.activations}"
        assert len(self.convs) == len(self.poolings), \
            f"convs and poolings must have same length: {self.convs} vs. {self.poolings}"
        for layer in self.convs:
            assert isinstance(layer, (list, tuple)) and len(layer) == 3, layer
            kernel_size = layer[1]
            if isinstance(kernel_size, int):
                assert kernel_size % 2 == 1, "VGGNLayerActFrontendV1 only supports odd kernel sizes"
        for layer in self.poolings:
            if isinstance(layer, (list, tuple)):
                assert len(layer) == 3, layer
            else:
                assert layer is None, layer
            kernel_size = None if layer is None else layer[0]
            if isinstance(kernel_size, int):
                assert kernel_size % 2 == 1, "VGGNLayerActFrontendV1 only supports odd kernel sizes"

    def __post__init__(self):
        super().__post_init__()
        self.check_valid()


@dataclass
class ModelConfig:
    specaug_config: Union[SpecaugConfig, SpecaugStftConfig]
    feature_extraction_config: ModelConfiguration
    frontend_config: ModelConfiguration
    frontend_config_class: str
    pos_emb_config: ConformerPosEmbConfig
    specaug_start_epoch: int
    label_target_size: int
    conformer_size: int
    num_layers: int
    num_heads: int
    ff_dim: int
    att_weights_dropout: float
    conv_dropout: float
    ff_dropout: float
    mhsa_dropout: float
    mhsa_with_bias: bool
    conv_kernel_size: int
    final_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    module_list: List[str]
    module_scales: List[float]
    aux_ctc_loss_layers: Optional[List[int]]
    aux_ctc_loss_scales: Optional[List[float]]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        feature_extraction_config_class = globals()[d["feature_extraction_config"]["module_class"] + "Config"]
        d["feature_extraction_config"] = feature_extraction_config_class.from_dict(d["feature_extraction_config"])
        frontend_config_class = globals()[d["frontend_config_class"]]
        d["frontend_config"] = frontend_config_class(**d["frontend_config"])
        specaug_config_class = SpecaugConfig
        if "fft_size" in d["specaug_config"] and "min_num_time" in d["specaug_config"]:
            specaug_config_class = SpecaugStftV2Config
        elif "fft_size" in d["specaug_config"]:
            specaug_config_class = SpecaugStftConfig
        d["specaug_config"] = specaug_config_class.from_dict(d["specaug_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        return ModelConfig(**d)
