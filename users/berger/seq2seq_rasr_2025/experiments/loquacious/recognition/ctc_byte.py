from dataclasses import dataclass, fields
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_experiments.common.setups.serialization import Collection

from ....data.loquacious import datasets as loquacious_datasets
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
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model: TrainedModel[ConformerCTCConfig],
    train_corpus_key: loquacious_datasets.TrainSet = "train.medium",
    variants: Optional[List[CTCRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = loquacious_datasets.EVAL_SETS

    results = []
    for variant in variants:
        results.extend(
            _run_single_variant(
                model=model,
                variant=variant,
                train_corpus_key=train_corpus_key,
                corpora=corpora,
            )
        )
    return results


def default_recog_variants() -> List[CTCRecogVariant]:
    return [
        default_offline_lexfree_recog_variant(),
        default_streaming_lexfree_recog_variant(),
    ]


def default_offline_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
    )


def default_streaming_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def _get_model_serializers(
    model: TrainedModel[ConformerCTCConfig],
    variant: CTCRecogVariant,
    train_corpus_key: loquacious_datasets.TrainSet,
) -> Collection:
    checkpoint = model.get_checkpoint(variant.epoch)
    prior_file = None
    if variant.prior_scale != 0.0:
        prior_file = compute_priors(
            prior_data_config=loquacious_datasets.get_prior_data(train_corpus_key=train_corpus_key),
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
    model: TrainedModel[ConformerCTCConfig],
    variant: CTCRecogVariant,
    train_corpus_key: loquacious_datasets.TrainSet,
    corpora: List[loquacious_datasets.EvalSet],
) -> List[RecogResult]:
    return run_single_byte_variant(
        model_descriptor=model.descriptor,
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(
            model=model,
            variant=variant,
            train_corpus_key=train_corpus_key,
        ),
        label_scorer_configs=_get_label_scorer_configs(),
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=None,
        variant=variant,
        corpora=corpora,
    )
