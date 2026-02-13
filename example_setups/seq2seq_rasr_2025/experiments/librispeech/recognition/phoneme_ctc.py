from dataclasses import dataclass, fields
from typing import List, Optional

from i6_experiments.common.setups.serialization import Collection

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from ....model_pipelines.common.recog import (
    OfflineRecogParameters,
    RecogResult,
    StreamingRecogParameters,
)
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCRecogConfig, ConformerCTCRecogModel
from ....model_pipelines.ctc.train import TrainedCTCModel
from .common import BaseRecogVariant, run_single_phoneme_variant


@dataclass
class CTCRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def _get_model_serializers(model: TrainedCTCModel, variant: CTCRecogVariant) -> Collection:
    checkpoint = model.get_checkpoint(variant.epoch)
    if variant.prior_scale != 0.0:
        prior_file = compute_priors(
            prior_data_config=librispeech_datasets.get_default_prior_data(),
            model_config=model.model_config,
            checkpoint=checkpoint,
        )
    return get_model_serializers(
        ConformerCTCRecogModel,
        ConformerCTCRecogConfig(
            **{f.name: getattr(model.model_config, f.name) for f in fields(model.model_config)},
            prior_file=prior_file if variant.prior_scale != 0.0 else None,
            prior_scale=variant.prior_scale,
            blank_penalty=variant.blank_penalty,
        ),
    )


def default_offline_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[512],
            score_thresholds=[12.0],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
            max_word_end_beam_size=16,
            word_end_score_threshold=0.5,
        ),
        prior_scale=0.2,
    )


def default_offline_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[512],
            score_thresholds=[14.0],
            word_lm_params=librispeech_lm.KazukiTrafoLmParams(scale=0.8),
            max_word_end_beam_size=16,
            word_end_score_threshold=0.5,
        ),
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=24),
        prior_scale=0.2,
    )


def default_streaming_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_streaming_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_end_score_threshold=0.5,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_recog_variants() -> List[CTCRecogVariant]:
    return [
        default_offline_4gram_recog_variant(),
        # default_offline_trafo_recog_variant(),
        default_streaming_4gram_recog_variant(),
    ]


def _run_single_variant(
    model: TrainedCTCModel, variant: CTCRecogVariant, corpora: List[librispeech_datasets.EvalSet]
) -> List[RecogResult]:
    return run_single_phoneme_variant(
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(model=model, variant=variant),
        label_scorer_configs=[get_no_op_label_scorer_config()],
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=None,
        variant=variant,
        corpora=corpora,
    )


def run(
    model: TrainedCTCModel,
    variants: Optional[List[CTCRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = librispeech_datasets.EVAL_SETS

    results = []

    for variant in variants:
        results.extend(_run_single_variant(model=model, variant=variant, corpora=corpora))
    return results
