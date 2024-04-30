"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass
import enum

import torch
from torch import nn
from typing import Callable, Optional, Type, Union

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
    ModelConfig as BaseCTCConfig,
    LogMelFeatureExtractionV1Config,
    VGG4LayerActFrontendV1Config_mod,
    SpecaugConfig
)

from ...rnnt.auxil.functional import TrainingStrategy

@dataclass
class ModelConfig(BaseCTCConfig):
    chunk_size: float
    carry_over_size: int
    lookahead_size: int
    online_model_scale: float

    training_strategy: Union[str, TrainingStrategy]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])

        enum_dict = {str(strat):strat for strat in TrainingStrategy}

        d["training_strategy"] = enum_dict[d["training_strategy"]]

        return ModelConfig(**d)
    
@dataclass
class ModelConfigNSplits(BaseCTCConfig):
    chunk_size: float
    carry_over_size: int
    lookahead_size: int
    online_model_scale: float
    num_splits: int

    training_strategy: Union[str, TrainingStrategy]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])

        enum_dict = {str(strat):strat for strat in TrainingStrategy}

        d["training_strategy"] = enum_dict[d["training_strategy"]]

        return ModelConfigNSplits(**d)