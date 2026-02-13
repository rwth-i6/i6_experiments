from dataclasses import dataclass, fields
from typing import List, Optional

from i6_experiments.common.setups.serialization import Collection

from ....data.loquacious import datasets as loquacious_datasets
from ....data.loquacious import lm as loquacious_lm
from ....data.loquacious.bpe import vocab_to_bpe_size
from ....data.loquacious.recog import LoquaciousTreeTimesyncRecogParams
from ....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from ....model_pipelines.common.recog import RecogResult, StreamingRecogParameters
from ....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCRecogConfig, ConformerCTCRecogModel
from ....model_pipelines.ctc.train import TrainedCTCModel
from .common import BaseRecogVariant, run_single_bpe_variant


@dataclass
class CTCRecogVariant(BaseRecogVariant):
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def _get_model_serializers(
    model: TrainedCTCModel, train_corpus_key: loquacious_datasets.TrainSet, variant: CTCRecogVariant
) -> Collection:
    checkpoint = model.get_checkpoint(variant.epoch)
    if variant.prior_scale != 0.0:
        prior_file = compute_priors(
            prior_data_config=loquacious_datasets.get_prior_data(train_corpus_key),
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


def default_offline_lexfree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
    )


def default_offline_tree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_tree",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
        ),
    )


def default_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_tree_4gram",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_lm_params=loquacious_lm.ArpaLmParams(scale=0.6),
            word_end_score_threshold=0.5,
        ),
        prior_scale=0.2,
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


def default_streaming_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="recog_streaming_tree_4gram",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_end_score_threshold=0.5,
            word_lm_params=loquacious_lm.ArpaLmParams(scale=0.6),
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_recog_variants() -> List[CTCRecogVariant]:
    return [
        default_offline_lexfree_recog_variant(),
        default_offline_tree_recog_variant(),
        default_offline_tree_4gram_recog_variant(),
        default_streaming_lexfree_recog_variant(),
        default_streaming_tree_4gram_recog_variant(),
    ]


def _run_single_variant(
    model: TrainedCTCModel,
    variant: CTCRecogVariant,
    train_corpus_key: loquacious_datasets.TrainSet,
    corpora: List[loquacious_datasets.EvalSet],
) -> List[RecogResult]:

    return run_single_bpe_variant(
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(model=model, train_corpus_key=train_corpus_key, variant=variant),
        label_scorer_configs=[get_no_op_label_scorer_config()],
        bpe_size=vocab_to_bpe_size(model.model_config.target_size - 1),
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=None,
        variant=variant,
        train_corpus_key=train_corpus_key,
        corpora=corpora,
    )


def run(
    model: TrainedCTCModel,
    train_corpus_key: loquacious_datasets.TrainSet,
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
            _run_single_variant(model=model, variant=variant, train_corpus_key=train_corpus_key, corpora=corpora)
        )
    return results
