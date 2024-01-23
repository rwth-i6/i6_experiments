"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Type, Union

from i6_models.assemblies.conformer.conformer_v1 import ConformerBlockV1Config, ConformerBlockV1
from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModuleFactoryV1, ModelConfiguration


@dataclass
class PredictorConfig(ModelConfiguration):
    symbol_embedding_dim: int
    emebdding_dropout: float
    num_lstm_layers: int
    lstm_hidden_dim: int
    lstm_dropout: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return PredictorConfig(**d)


@dataclass
class HubertConfig(ModelConfiguration):
    name: str
    finetune_layer: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return HubertConfig(**d)


@dataclass
class ModelConfig:
    predictor_config: PredictorConfig
    specauc_start_epoch: int
    label_target_size: int
    final_dropout: float
    joiner_dim: int
    joiner_activation: str
    joiner_dropout: float
    hubert_cfg: HubertConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["predictor_config"] = PredictorConfig.from_dict(d["predictor_config"])
        d["hubert_cfg"] = HubertConfig.from_dict(d["hubert_cfg"])
        return ModelConfig(**d)
