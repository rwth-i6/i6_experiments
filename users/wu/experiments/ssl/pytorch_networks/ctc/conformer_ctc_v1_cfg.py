"""Config dataclass for the CTC model (``conformer_ctc_v1.Model``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ..common.conformer import ConformerEncoderConfig


@dataclass
class SpecaugConfig:
    repeat_per_n_frames: int
    max_dim_time: int
    num_repeat_feat: int
    max_dim_feat: int


@dataclass
class CTCConfig:
    feature_extraction_config: LogMelFeatureExtractionV1Config
    encoder_config: ConformerEncoderConfig
    # same fixed global (corpus-level) log-mel normalization as SSL pretraining, so the encoder sees
    # the identical input distribution -> the pretrained weights transfer.
    global_mean: List[float]
    global_std: List[float]
    specaug_config: SpecaugConfig
    specaug_start_step: int  # SpecAugment enabled from this global train step onward (train only)
    # Auxiliary CTC: extra CTC head(s) on intermediate encoder layers (1-indexed) with the given loss
    # scales, summed into the loss alongside the final-layer head (implicit scale 1.0). None = no aux.
    aux_ctc_layers: Optional[List[int]] = None
    aux_ctc_scales: Optional[List[float]] = None

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["encoder_config"] = ConformerEncoderConfig.from_dict(d["encoder_config"])
        d["specaug_config"] = SpecaugConfig(**d["specaug_config"])
        return CTCConfig(**d)
