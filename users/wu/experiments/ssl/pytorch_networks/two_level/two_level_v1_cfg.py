"""Config dataclass for the two-level BEST-RQ + CIF SSL model (``two_level_v1.Model``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ..common.conformer import ConformerEncoderConfig, HighConformerConfig


@dataclass
class TwoLevelConfig:
    # ---- frozen lower encoder (log-mel + global-norm + 9-layer rel-pos Conformer) ----
    # Built and named exactly like the BEST-RQ model (``self.feature_extraction`` / ``self.encoder``)
    # so the pretrained checkpoint preloads 1:1. encoder_config.num_layers MUST be 9 -> its state_dict
    # keys (encoder.module_list.0..8) are a subset of the 12-layer BEST-RQ ckpt; blocks 9-11 + quantizer
    # + heads in the ckpt are extra and skipped via preload ignore_missing.
    feature_extraction_config: LogMelFeatureExtractionV1Config
    encoder_config: ConformerEncoderConfig
    global_mean: List[float]
    global_std: List[float]
    lower_layer_index: int  # 0-based block whose output feeds CIF (8 == output after the 9th layer)

    # ---- CIF predictor + pooling ----
    cif_alpha_kernel_size: int  # temporal conv kernel for the alpha predictor (5 or 7)
    target_rate_hz: float       # desired CIF token rate (12.5 == 80 ms, 8.333 == 120 ms at 25 Hz)
    frame_rate_hz: float        # lower-encoder frame rate (DERIVED = 25.0; asserted in the model)
    lambda_qty: float           # weight of the quantity loss in the total objective

    # ---- frozen high-level k-means target codebook (centroids loaded from the ``codebook`` net-arg) ----
    num_clusters: int           # codebook size (128 / 256); also the prediction-head output dim

    # ---- high/global Conformer over CIF tokens ----
    high_encoder_config: HighConformerConfig

    # ---- span masking over the CIF-token sequence (mask_prob is a span-START prob; realized coverage
    # ~ 1-(1-p)^L and is logged, not assumed) ----
    mask_prob: float
    mask_length: int
    min_masks: int

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["encoder_config"] = ConformerEncoderConfig.from_dict(d["encoder_config"])
        d["high_encoder_config"] = HighConformerConfig.from_dict(d["high_encoder_config"])
        return TwoLevelConfig(**d)
