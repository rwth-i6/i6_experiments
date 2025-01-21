import torch
from dataclasses import dataclass, asdict
from typing import Union, Callable

# FastConformer relevant imports
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config
from .generic_frontend_v2 import GenericFrontendV2Config

# complete Transducer config
from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v9_cfg import (
    SpecaugConfig,
    PredictorConfig,
    VGG4LayerActFrontendV1Config_mod,
)
from .streaming_conformer_cfg import ModelConfig as ModelConfigStreaming

from ..conformer_2.conformer_vrnn import VRNNV1Config



@dataclass
class ModelConfig(ModelConfigStreaming):
    lookahead_size: float

    # unified model settings
    online_model_scale: float

    # vrnn settings
    vrnn_dropout: float
    vrnn_activation: Union[str, Callable[[torch.Tensor], torch.Tensor]]

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

        if d["vrnn_activation"] == "silu":
            d["vrnn_activation"] = torch.nn.functional.silu
        else:
            d["vrnn_activation"] = torch.nn.functional.relu

        return ModelConfig(**d)