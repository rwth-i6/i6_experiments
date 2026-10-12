"""Pure helpers for label-free alignment evaluation."""

from .metrics import (AgreementResult, BoundaryDisagreement, DeviationStats, FailureCounts,
                      FlaggedBoundary, OffsetStats, PerturbationResult, TransformResult, aggregate_agreement,
                      aggregate_perturbation, boundary_disagreement, bootstrap_median_ci,
                      deviation_stats, format_stats_table, leave_one_out_deviations_ms,
                      offset_stats, onset_failure_reason, perturbation_deviations_ms, word_offsets_ms)
from .onsets import mfa_words_to_onsets, path_to_token_onsets, path_to_word_spans, token_onsets_to_words
from .transforms import (Transform, additive_noise, default_transforms, gain, resample_roundtrip,
                         silence_pad_start, speed)

__all__ = [
    "AgreementResult", "BoundaryDisagreement", "DeviationStats", "FailureCounts",
    "FlaggedBoundary", "OffsetStats", "PerturbationResult", "TransformResult", "aggregate_agreement",
    "aggregate_perturbation", "boundary_disagreement", "bootstrap_median_ci",
    "deviation_stats", "format_stats_table", "leave_one_out_deviations_ms",
    "offset_stats", "onset_failure_reason", "perturbation_deviations_ms", "word_offsets_ms",
    "mfa_words_to_onsets", "path_to_token_onsets", "path_to_word_spans", "token_onsets_to_words",
    "Transform", "additive_noise", "default_transforms", "gain", "resample_roundtrip",
    "silence_pad_start", "speed",
]
