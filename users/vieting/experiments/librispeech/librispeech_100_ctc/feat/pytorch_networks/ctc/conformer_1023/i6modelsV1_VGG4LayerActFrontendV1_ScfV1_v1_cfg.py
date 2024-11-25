"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Type

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration
from .i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import SpecaugConfig, VGG4LayerActFrontendV1Config_mod
from .feature_extraction import SupervisedConvolutionalFeatureExtractionV1Config

@dataclass
class ModelConfig:
    feature_extraction_config: SupervisedConvolutionalFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    specaug_start_epoch: int
    feature_training_start_epoch: int
    feature_training_end_epoch: int
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

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = SupervisedConvolutionalFeatureExtractionV1Config(
            **d["feature_extraction_config"]
        )
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return ModelConfig(**d)
