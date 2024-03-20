"""
Config for the base Parakeet Models v1
"""

from dataclasses import dataclass
from typing import Optional, Union, List
from i6_models.config import ModelConfiguration

@dataclass
class ParakeetConfig(ModelConfiguration):
    name: str
    finetune_layer: Optional[Union[int, bool]]
    keep_layers: Optional[Union[int, List]]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return ParakeetConfig(**d)


@dataclass
class ModelConfig:
    label_target_size: int
    final_dropout: float
    parakeet_config: ParakeetConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["parakeet_config"] = ParakeetConfig.from_dict(d["parakeet_config"])
        return ModelConfig(**d)
