"""Config dataclass for the BEST-RQ SSL model (``best_rq_v1.Model``)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from i6_models.primitives.feature_extraction import LogMelFeatureExtractionV1Config

from ..common.conformer import ConformerEncoderConfig


@dataclass
class BestRQConfig:
    # feature extraction (log-mel) and the shared conformer encoder
    feature_extraction_config: LogMelFeatureExtractionV1Config
    encoder_config: ConformerEncoderConfig

    # fixed corpus-level (global) log-mel normalization, per feature [F]
    global_mean: List[float]
    global_std: List[float]

    # BEST-RQ quantizer (frozen, multi-codebook)
    stack_size: int          # frames stacked per target == encoder time subsampling (4)
    codebook_dim: int        # projected/code vector dim (16)
    vocab_size: int          # codes per codebook (8192)
    num_codebooks: int       # number of independent codebooks N (4)
    quantizer_seed: int      # fixed seed for the frozen projection/codebook

    # masking (at the subsampled / encoder-output frame rate)
    mask_prob: float         # per-frame span-start probability (~0.04 -> ~34% coverage with L=10)
    mask_length: int         # span length in subsampled frames (10 == 400ms at 40ms/frame)
    min_masks: int           # minimum spans forced per sequence
    noise_std: float         # std of the N(0, std) noise that replaces masked input frames (0.1)

    @classmethod
    def from_dict(cls, d):
        d = d.copy()
        d["feature_extraction_config"] = LogMelFeatureExtractionV1Config(**d["feature_extraction_config"])
        d["encoder_config"] = ConformerEncoderConfig.from_dict(d["encoder_config"])
        return BestRQConfig(**d)
