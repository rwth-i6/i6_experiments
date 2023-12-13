"""
Config for the base CTC models v4, including specaug start time
"""

from dataclasses import dataclass

from i6_models.config import ModelConfiguration

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
    specauc_start_epoch: int
    label_target_size: int
    final_dropout: float
    whisper_config: WhisperConfig

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["whisper_config"] = WhisperConfig.from_dict(d["whisper_config"])
        return ModelConfig(**d)
