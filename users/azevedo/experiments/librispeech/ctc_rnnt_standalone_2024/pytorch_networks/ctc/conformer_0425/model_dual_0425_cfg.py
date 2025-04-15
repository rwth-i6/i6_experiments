from dataclasses import dataclass
from typing import Literal, Optional, Type, Union, List

from ...rnnt.conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
    SpecaugConfig,
    ConformerPosEmbConfig,
)

from ..conformer_1023.i6modelsV1_VGG4LayerActFrontendV1_v6_cfg import (
    LogMelFeatureExtractionV1Config,
    VGG4LayerActFrontendV1Config,
    VGG4LayerActFrontendV1Config_mod,
    SpecaugConfig
)

from ...rnnt.auxil.functional import TrainingStrategy


@dataclass
class ModelConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    specaug_config: SpecaugConfig
    pos_emb_config: ConformerPosEmbConfig
    specauc_start_epoch: int
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

    fastemit_lambda: Optional[float]
    chunk_size: float
    lookahead_size: int
    carry_over_size: int
    online_model_scale: float
    training_strategy: Union[str, TrainingStrategy]
    dual_mode: bool

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])

        enum_dict = {str(strat): strat for strat in TrainingStrategy}
        d["training_strategy"] = enum_dict[d["training_strategy"]]

        return ModelConfig(**d)
