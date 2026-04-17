from dataclasses import dataclass, fields
from typing import List, Optional

from i6_core.rasr import RasrConfig
from i6_experiments.common.setups.serialization import Collection

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from ....model_pipelines.common.recog import OfflineRecogParameters, RecogResult, StreamingRecogParameters
from ....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.common.train import TrainedModel
from ....model_pipelines.ctc.prior import compute_priors
from ....model_pipelines.ctc.pytorch_modules import ConformerCTCConfig, ConformerCTCRecogConfig, ConformerCTCRecogModel
from .common import BaseRecogVariant, run_single_bpe_variant


@dataclass
class CTCRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    bpe_trafo_lm_scale: float = 0.0
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
        default_offline_lexfree_lstm_recog_variant(),
        default_offline_lexfree_trafo_recog_variant(),
        default_offline_tree_recog_variant(),
        default_offline_tree_4gram_recog_variant(),
        default_offline_tree_lstm_recog_variant(),
        default_offline_tree_lstm_4gram_recog_variant(),
        default_offline_tree_trafo_recog_variant(),
        default_offline_tree_trafo_recog_variant_gpu(),
        default_streaming_lexfree_recog_variant(),
        default_streaming_tree_4gram_recog_variant(),
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


def default_offline_lexfree_lstm_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree_bpe-LSTM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0, 12.0],
            max_beam_sizes=[2048, 256],
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_lexfree_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="lexfree_bpe-TrafoLM",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0, 12.0],
            max_beam_sizes=[2048, 256],
        ),
        search_mode_params=OfflineRecogParameters(mem_rqmt=24),
        prior_scale=0.2,
        bpe_trafo_lm_scale=0.8,
    )


def default_offline_tree_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[6.0],
            max_beam_sizes=[8],
            word_end_score_threshold=0.0,
            max_word_end_beam_size=1,
        ),
    )


def default_offline_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
            score_thresholds=[12.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
        ),
        prior_scale=0.2,
    )


def default_offline_tree_lstm_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[12.0, 10.0],
            max_beam_sizes=[128, 64],
            word_end_score_threshold=0.6,
            max_word_end_beam_size=16,
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.8,
    )


def default_offline_tree_lstm_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_4gram_bpe-LSTM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[10.0, 10.0],
            max_beam_sizes=[128, 64],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=32,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.2),
        ),
        prior_scale=0.2,
        bpe_lstm_lm_scale=0.6,
    )


def default_offline_tree_trafo_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8),
        ),
        search_mode_params=OfflineRecogParameters(),
        prior_scale=0.2,
    )


def default_offline_tree_trafo_recog_variant_gpu() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="tree_trafoLM_gpu",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8, use_gpu=True, use_kv_cache=False),
        ),
        search_mode_params=OfflineRecogParameters(gpu_mem_rqmt=24),
        prior_scale=0.2,
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


def default_streaming_tree_4gram_recog_variant() -> CTCRecogVariant:
    return CTCRecogVariant(
        descriptor="streaming_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0],
            max_beam_sizes=[1024],
            word_end_score_threshold=0.5,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        search_mode_params=StreamingRecogParameters(encoder_frame_shift_seconds=0.04),
    )


def _get_model_serializers(model: TrainedModel[ConformerCTCConfig], variant: CTCRecogVariant) -> Collection:
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


def _get_label_scorer_configs(model: TrainedModel[ConformerCTCConfig], variant: CTCRecogVariant) -> List[RasrConfig]:
    bpe_size = vocab_to_bpe_size(model.model_config.target_size - 1)

    label_scorer_configs = [get_no_op_label_scorer_config()]
    if variant.bpe_lstm_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_lstm_label_scorer_config(
                bpe_size=bpe_size,
                scale=variant.bpe_lstm_lm_scale,
                use_gpu=variant.search_mode_params.gpu_mem_rqmt > 0,
            )
        )
    if variant.bpe_trafo_lm_scale != 0.0:
        label_scorer_configs.append(
            librispeech_lm.get_bpe_transformer_label_scorer_config(
                bpe_size=bpe_size,
                use_gpu=variant.search_mode_params.gpu_mem_rqmt > 0,
                scale=variant.bpe_trafo_lm_scale,
            )
        )

    return label_scorer_configs


def _run_single_variant(
    model: TrainedModel[ConformerCTCConfig], variant: CTCRecogVariant, corpora: List[librispeech_datasets.EvalSet]
) -> List[RecogResult]:

    return run_single_bpe_variant(
        model_descriptor=model.descriptor,
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=_get_model_serializers(model=model, variant=variant),
        label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
        bpe_size=vocab_to_bpe_size(model.model_config.target_size - 1),
        blank_index=model.model_config.target_size - 1,
        sentence_end_index=0 if variant.bpe_lstm_lm_scale != 0 else None,
        variant=variant,
        corpora=corpora,
    )
