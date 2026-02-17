from dataclasses import dataclass
from typing import List, Optional

from i6_core.rasr import RasrConfig

from ....model_pipelines.ffnn_transducer.label_scorer_config import get_ffnn_transducer_label_scorer_config
from ....model_pipelines.ffnn_transducer.pytorch_modules import FFNNTransducerEncoder

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.common.recog import (
    OfflineRecogParameters,
    RecogResult,
    StreamingRecogParameters,
)
from ....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.ffnn_transducer.train import TrainedFFNNTransducerModel
from .common import BaseRecogVariant, run_single_bpe_variant


@dataclass
class TransducerRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    ilm_scale: float = 0.0
    blank_penalty: float = 0.0


def _get_label_scorer_configs(model: TrainedFFNNTransducerModel, variant: TransducerRecogVariant) -> List[RasrConfig]:
    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)
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
    if variant.bpe_lstm_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_lstm_label_scorer_config(
                bpe_size=bpe_size,
                scale=variant.bpe_lstm_lm_scale,
                use_gpu=use_gpu,
            )
        )

    return label_scorer_configs


def default_offline_lexfree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
    )


def default_offline_lexfree_lstm_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_lexfree_bpe-LSTM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[30, 20],
            score_thresholds=[8.0, 8.0],
        ),
        ilm_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256],
            score_thresholds=[14.0],
        ),
    )


def default_offline_tree_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256],
            score_thresholds=[14.0],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
        ),
        ilm_scale=0.2,
    )


def default_offline_tree_lstm_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[40, 20],
            score_thresholds=[8.0, 8.0],
        ),
        ilm_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_lstm_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_4gram_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[150, 100],
            score_thresholds=[10.0, 10.0],
            word_end_score_threshold=1.0,
            max_word_end_beam_size=30,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.2),
        ),
        ilm_scale=0.0,
        bpe_lstm_lm_scale=0.6,
    )


def default_offline_tree_trafo_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_tree_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[512],
            score_thresholds=[14.0],
            word_lm_params=librispeech_lm.KazukiTrafoLmParams(scale=0.8),
            max_word_end_beam_size=16,
            word_end_score_threshold=0.5,
        ),
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=24),
        ilm_scale=0.2,
    )


def default_streaming_lexfree_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[256],
            score_thresholds=[14.0],
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def default_streaming_tree_4gram_recog_variant() -> TransducerRecogVariant:
    return TransducerRecogVariant(
        descriptor="recog_streaming_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=False,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
            max_word_end_beam_size=16,
            word_end_score_threshold=0.5,
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
        ilm_scale=0.2,
    )


def default_recog_variants() -> List[TransducerRecogVariant]:
    return [
        default_offline_lexfree_recog_variant(),
        default_offline_lexfree_lstm_recog_variant(),
        default_offline_tree_recog_variant(),
        default_offline_tree_4gram_recog_variant(),
        default_offline_tree_lstm_recog_variant(),
        default_offline_tree_lstm_4gram_recog_variant(),
        # default_offline_tree_trafo_recog_variant(),
        default_streaming_lexfree_recog_variant(),
        default_streaming_tree_4gram_recog_variant(),
    ]


def _run_single_variant(
    model: TrainedFFNNTransducerModel, variant: TransducerRecogVariant, corpora: List[librispeech_datasets.EvalSet]
) -> List[RecogResult]:
    return run_single_bpe_variant(
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=get_model_serializers(FFNNTransducerEncoder, model.model_config),
        label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
        bpe_size=vocab_to_bpe_size(model.model_config.target_size - 1),
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 else None,
        variant=variant,
        corpora=corpora,
    )


def run(
    model: TrainedFFNNTransducerModel,
    variants: Optional[List[TransducerRecogVariant]] = None,
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
