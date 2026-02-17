from dataclasses import dataclass
from typing import List, Optional

from i6_core.rasr import RasrConfig

from ....data.loquacious import datasets as loquacious_datasets
from ....data.loquacious import lm as loquacious_lm
from ....data.loquacious.recog import LoquaciousTreeTimesyncRecogParams
from ....model_pipelines.common.recog import RecogResult, StreamingRecogParameters
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.ffnn_transducer.label_scorer_config import get_ffnn_transducer_label_scorer_config
from ....model_pipelines.ffnn_transducer.pytorch_modules import FFNNTransducerEncoder
from ....model_pipelines.ffnn_transducer.train import TrainedFFNNTransducerModel
from .common import BaseRecogVariant, run_single_phoneme_variant


@dataclass
class TransducerRecogVariant(BaseRecogVariant):
    ilm_scale: float = 0.0
    blank_penalty: float = 0.0


def _get_label_scorer_configs(model: TrainedFFNNTransducerModel, variant: TransducerRecogVariant) -> List[RasrConfig]:
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0

    label_scorer_configs = [
        get_ffnn_transducer_label_scorer_config(
            model_config=model.model_config,
            checkpoint=model.get_checkpoint(variant.epoch),
            ilm_scale=variant.ilm_scale,
            blank_penalty=variant.blank_penalty,
            use_gpu=use_gpu,
        )
    ]

    return label_scorer_configs


def default_offline_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_4gram",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256],
            score_thresholds=[14.0],
            word_lm_params=loquacious_lm.ArpaLmParams(scale=0.3),
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
        ),
        ilm_scale=0.2,
    )


def default_streaming_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_streaming_4gram",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_lm_params=loquacious_lm.ArpaLmParams(scale=0.3),
            max_word_end_beam_size=16,
            word_end_score_threshold=0.5,
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
        ilm_scale=0.2,
    )


def default_recog_variants() -> List[TransducerRecogVariant]:
    return [
        default_offline_4gram_recog_variant(),
        default_streaming_4gram_recog_variant(),
    ]


def _run_single_variant(
    model: TrainedFFNNTransducerModel,
    variant: TransducerRecogVariant,
    corpora: List[loquacious_datasets.EvalSet],
) -> List[RecogResult]:
    return run_single_phoneme_variant(
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=get_model_serializers(FFNNTransducerEncoder, model.model_config),
        label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=None,
        variant=variant,
        corpora=corpora,
    )


def run(
    model: TrainedFFNNTransducerModel,
    variants: Optional[List[TransducerRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
) -> List[RecogResult]:
    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = loquacious_datasets.EVAL_SETS

    results = []

    for variant in variants:
        results.extend(_run_single_variant(model=model, variant=variant, corpora=corpora))
    return results
