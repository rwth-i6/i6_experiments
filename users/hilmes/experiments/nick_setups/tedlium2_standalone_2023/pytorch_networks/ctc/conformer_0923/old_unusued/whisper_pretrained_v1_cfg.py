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
class WhisperConfig(ModelConfiguration):
    name: str
    just_encoder: bool
    finetune_layer: int
    split_seq: bool
    dropout: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return WhisperConfig(**d)


@dataclass
class ModelConfig:
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    specauc_start_epoch: int
    label_target_size: int
    final_dropout: float
    whisper_config: WhisperConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["whisper_config"] = WhisperConfig.from_dict(d["whisper_config"])
        return ModelConfig(**d)
