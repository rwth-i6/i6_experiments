"""
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, List, Literal, Optional, Union

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from .i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
    VGG4LayerActFrontendV1Config_mod,
    SpecaugConfig,
    ConformerPosEmbConfig,
)


@dataclass
class ModelConfig:
    feature_extraction_config: ModelConfiguration
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
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
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        return ModelConfig(**d)
