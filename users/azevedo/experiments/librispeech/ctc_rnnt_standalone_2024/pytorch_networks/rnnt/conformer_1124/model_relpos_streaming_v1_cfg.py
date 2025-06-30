from ..conformer_0924.i6models_relposV1_VGG4LayerActFrontendV1_v1_cfg import (
    PredictorConfig,
    SpecaugConfig,
    ConformerPosEmbConfig,
    VGG4LayerActFrontendV1Config_mod
)
from dataclasses import dataclass

from typing import List, Literal, Optional, Union

from i6_models.parts.frontend.vgg_act import VGG4LayerActFrontendV1Config
from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ..auxil.functional import TrainingStrategy


@dataclass
class ModelConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    frontend_config: VGG4LayerActFrontendV1Config
    predictor_config: PredictorConfig
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
    joiner_dim: int
    joiner_activation: str
    joiner_dropout: float
    dropout_broadcast_axes: Optional[Literal["B", "T", "BT"]]
    module_list: List[str]
    module_scales: List[float]
    aux_ctc_loss_layers: Optional[List[int]]
    aux_ctc_loss_scales: Optional[List[float]]
    ctc_output_loss: float

    fastemit_lambda: Optional[float]
    chunk_size: float
    lookahead_size: int
    carry_over_size: int
    online_model_scale: float
    training_strategy: Union[str, TrainingStrategy]

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["frontend_config"] = VGG4LayerActFrontendV1Config_mod.from_dict(d["frontend_config"])
        d["specaug_config"] = SpecaugConfig.from_dict(d["specaug_config"])
        d["pos_emb_config"] = ConformerPosEmbConfig(**d["pos_emb_config"])
        d["predictor_config"] = PredictorConfig.from_dict(d["predictor_config"])

        enum_dict = {str(strat): strat for strat in TrainingStrategy}
        d["training_strategy"] = enum_dict[d["training_strategy"]]

        return ModelConfig(**d)
