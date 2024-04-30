"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Type, Union

from .i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
    ModelConfig as BaseCTCConfig,
    LogMelFeatureExtractionV1Config,
    VGG4LayerActFrontendV1Config_mod,
    SpecaugConfig
)


@dataclass
class ModelConfig(BaseCTCConfig):
    chunk_size: float
    lookahead_size: int
    online_model_scale: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])

        return ModelConfig(**d)


class ModelConfigCOV2(BaseCTCConfig):
    chunk_size: float
    carry_over_size: int
    lookahead_size: int
    online_model_scale: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])

        return ModelConfig(**d)
