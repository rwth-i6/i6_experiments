from dataclasses import dataclass

# FastConformer relevant imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from ..streaming_conformer.generic_frontend_v2 import GenericFrontendV2Config

# complete Transducer config
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
    SpecaugConfig,
    PredictorConfig,
    VGG4LayerActFrontendV1Config_mod,
)

from ..streaming_conformer.streaming_conformer_cfg import ModelConfig as ModelConfigStreaming

@dataclass
class ModelConfig(ModelConfigStreaming):
    carry_over_size: int

    online_model_scale: float

    avg_fut_latency: float

    aggr_num_layers: int
    aggr_lstm_dim: int
    aggr_lstm_dropout: float
    aggr_ff_dim: int
    aggr_weights_dropout: float
    aggr_attn_heads: int
    aggr_bias: bool
    aggr_noise_std: float

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])

        if d.get("use_vgg", False):
            print(d["frontend_config"])
            d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        else:
            d["frontend_config"] = GenericFrontendV2Config.from_dict(d["frontend_config"])

        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["predictor_config"] = PredictorConfig.from_dict(d["predictor_config"])

        return ModelConfig(**d)