"""
Config for the base Hubert tuning
"""

from dataclasses import dataclass
from typing import Optional, Union, List

from i6_models.config import ModelConfiguration

@dataclass
class ModelConfig:
    label_target_size: int
    final_dropout: float
    model_name: str
    finetune_layer: Optional[Union[int, bool]]
    keep_layers: Optional[Union[int, List]]
    downsample_factor: Optional[int]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        return ModelConfig(**d)
