from dataclasses import dataclass

from typing import Union

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
from i6_models.assemblies.conformer.conformer_v1 import ConformerEncoderV1Config

from ..auxil.functional import TrainingStrategy


@dataclass
class ModelConfig(ModelConfigStreaming):
    lookahead_size: int
    carry_over_size: int

    online_model_scale: float

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

@dataclass
class ModelConfigPrefrontendLAH(ModelConfigStreaming):
    lookahead_size: float
    carry_over_size: int
    online_model_scale: float

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

        return ModelConfigPrefrontendLAH(**d)
    


@dataclass
class ModelConfigNSplits(ModelConfigStreaming):
    lookahead_size: int
    carry_over_size: int
    num_splits: int

    online_model_scale: float

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

        return ModelConfigNSplits(**d)
    


@dataclass
class ModelConfigV2(ModelConfigStreaming):
    lookahead_size: int
    carry_over_size: int

    online_model_scale: float

    training_strategy: Union[str, TrainingStrategy]

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

        enum_dict = {str(strat):strat for strat in TrainingStrategy}

        d["training_strategy"] = enum_dict[d["training_strategy"]]

        return ModelConfigV2(**d)