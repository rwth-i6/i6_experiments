from dataclasses import dataclass, fields
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_experiments.common.setups.serialization import Collection

from ....data.librispeech import datasets as librispeech_datasets
from ....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from ....model_pipelines.common.recog import RecogResult, StreamingRecogParameters
from ....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.common.train import TrainedModel
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCConfig, ConformerCTCRecogConfig, ConformerCTCRecogModel
from .common import BaseRecogVariant, run_single_byte_variant


@dataclass
class CTCRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model: TrainedModel[ConformerCTCConfig],
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


def default_recog_variants() -> List[CTCRecogVariant]:
    return [
        default_offline_lexfree_recog_variant(),
        default_streaming_lexfree_recog_variant(),
    ]


def default_offline_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
    )


def default_streaming_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def _get_model_serializers(model: TrainedModel[ConformerCTCConfig], variant: CTCRecogVariant) -> Collection:
    checkpoint = model.get_checkpoint(variant.epoch)
    prior_file = None
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
            prior_file=prior_file,
            prior_scale=variant.prior_scale,
            blank_penalty=variant.blank_penalty,
        ),
    )


def _get_label_scorer_configs() -> List[RasrConfig]:
    return [get_no_op_label_scorer_config()]


def _run_single_variant(
    model: TrainedModel[ConformerCTCConfig], variant: CTCRecogVariant, corpora: List[librispeech_datasets.EvalSet]
) -> List[RecogResult]:
    return run_single_byte_variant(
        model_descriptor=model.descriptor,
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(model=model, variant=variant),
        label_scorer_configs=_get_label_scorer_configs(),
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=None,
        variant=variant,
        corpora=corpora,
    )
