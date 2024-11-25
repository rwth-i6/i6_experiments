from typing import List
from dataclasses import dataclass

from i6_models.parts.blstm import BlstmEncoderV1Config

from i6_models.config import ModelConfiguration
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config


@dataclass(kw_only=True)
class BlstmPoolingEncoderConfig(BlstmEncoderV1Config):
    # Layers after which to insert a max pooling layer
    pooling_layer_positions: List[int]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        pooling_layer_positions = d.pop("pooling_layer_positions")
        return cls(**d, pooling_layer_positions=pooling_layer_positions)


@dataclass
class SpecaugConfig(ModelConfiguration):
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int


@dataclass
class BlstmModelConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    # TODO: terminology frontend
    frontend_config: BlstmPoolingEncoderConfig
    specaug_config: SpecaugConfig
    specaug_start_epoch: int
    label_target_size: int
    final_dropout: float
    fsa_config_path: str
    tdp_scale: float
    am_scale: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = BlstmPoolingEncoderConfig.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return BlstmModelConfig(**d)
