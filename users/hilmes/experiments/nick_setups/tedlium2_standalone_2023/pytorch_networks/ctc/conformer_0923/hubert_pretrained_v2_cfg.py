"""
Config for the base Hubert Models v2, including specaug start time
adds the option for keep layers
"""

from dataclasses import dataclass

import torch
from torch import nn
from typing import Callable, Optional, Type, Union, List

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
class HubertConfig(ModelConfiguration):
    name: str
    finetune_layer: Optional[Union[int, bool]]
    keep_layers: Optional[Union[int, List]]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return HubertConfig(**d)


@dataclass
class ModelConfig:
    specauc_start_epoch: int
    label_target_size: int
    final_dropout: float
    hubert_cfg: HubertConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["hubert_cfg"] = HubertConfig.from_dict(d["hubert_cfg"])
        return ModelConfig(**d)
