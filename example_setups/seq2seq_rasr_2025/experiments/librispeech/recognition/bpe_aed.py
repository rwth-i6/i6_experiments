from dataclasses import dataclass
from typing import List, Optional

from i6_core.rasr import RasrConfig

from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import vocab_to_bpe_size
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.aed.label_scorer_config import (
    get_aed_label_scorer_config,
    get_ctc_label_scorer_config,
    get_ctc_prefix_label_scorer_config,
)
from ....model_pipelines.aed.pytorch_modules import AEDEncoder
from ....model_pipelines.aed.train import TrainedAEDModel
from ....model_pipelines.common.recog import (
    RecogResult,
)
from ....model_pipelines.common.recog_rasr_config import (
    LexiconfreeLabelsyncRecogParams,
    LexiconfreeTimesyncRecogParams,
)
from ....model_pipelines.common.serializers import get_model_serializers
from .common import BaseRecogVariant, run_single_bpe_variant


@dataclass
class AEDRecogVariant(BaseRecogVariant):
    epoch: Optional[int] = None
    bpe_lstm_lm_scale: float = 0.0
    ctc_score_scale: float = 0.0


def default_lexfree_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            max_beam_sizes=[8],
            length_norm_scale=1.2,
        ),
    )


def default_lexfree_lstm_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree_bpe-LSTM",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            max_beam_sizes=[16, 8],
            length_norm_scale=1.2,
        ),
        bpe_lstm_lm_scale=0.6,
    )


def default_lexfree_aed_ctc_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree_aed+ctc_labelsync",
        search_algorithm_params=LexiconfreeLabelsyncRecogParams(
            max_beam_sizes=[16, 8],
            length_norm_scale=1.2,
        ),
        ctc_score_scale=0.3,
    )


def default_lexfree_aed_ctc_timesync_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_lexfree_aed+ctc_timesync",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            max_beam_sizes=[16, 8],
            collapse_repeated_labels=True,
        ),
        ctc_score_scale=0.3,
    )


def default_tree_aed_ctc_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_tree_aed+ctc",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16, 8],
        ),
        ctc_score_scale=0.3,
    )


def default_tree_aed_ctc_4gram_recog_variant() -> AEDRecogVariant:
    return AEDRecogVariant(
        descriptor="recog_tree_aed+ctc_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16, 8],
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
        ),
        ctc_score_scale=0.3,
    )


def default_recog_variants() -> List[AEDRecogVariant]:
    return [
        default_lexfree_recog_variant(),
        default_lexfree_lstm_recog_variant(),
        default_lexfree_aed_ctc_recog_variant(),
        default_lexfree_aed_ctc_timesync_recog_variant(),
        default_tree_aed_ctc_recog_variant(),
        default_tree_aed_ctc_4gram_recog_variant(),
    ]


def _get_label_scorer_configs(model: TrainedAEDModel, variant: AEDRecogVariant) -> List[RasrConfig]:
    checkpoint = model.get_checkpoint(variant.epoch)
    bpe_size = vocab_to_bpe_size(model.model_config.label_target_size)
    use_gpu = variant.search_mode_params.gpu_mem_rqmt > 0
    labelsync = isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)

    aed_label_scorer_config = get_aed_label_scorer_config(
        model_config=model.model_config,
        checkpoint=checkpoint,
        use_gpu=use_gpu,
        scale=1.0 - variant.ctc_score_scale,
    )

    ctc_label_scorer_config = None
    if variant.ctc_score_scale != 0.0:
        if labelsync:
            ctc_label_scorer_config = get_ctc_prefix_label_scorer_config(
                model_config=model.model_config,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                use_gpu=use_gpu,
            )
        else:
            ctc_label_scorer_config = get_ctc_label_scorer_config(
                model_config=model.model_config,
                checkpoint=checkpoint,
                scale=variant.ctc_score_scale,
                use_gpu=use_gpu,
            )

    lstm_lm_label_scorer_config = None
    if variant.bpe_lstm_lm_scale != 0.0:
        lstm_lm_label_scorer_config = librispeech_lm.get_bpe_lstm_label_scorer_config(
            bpe_size=bpe_size,
            scale=variant.bpe_lstm_lm_scale,
            use_gpu=use_gpu,
        )

    if labelsync:
        return list(filter(None, [aed_label_scorer_config, ctc_label_scorer_config, lstm_lm_label_scorer_config]))
    else:
        assert ctc_label_scorer_config is not None
        return list(filter(None, [ctc_label_scorer_config, aed_label_scorer_config, lstm_lm_label_scorer_config]))


def _run_single_variant(
    model: TrainedAEDModel, variant: AEDRecogVariant, corpora: List[librispeech_datasets.EvalSet]
) -> List[RecogResult]:
    if isinstance(
        variant.search_algorithm_params, (LexiconfreeTimesyncRecogParams, LibrispeechTreeTimesyncRecogParams)
    ):
        blank_index = model.model_config.label_target_size
    else:
        blank_index = None

    return run_single_bpe_variant(
        checkpoint=model.get_checkpoint(variant.epoch),
        encoder_serializers=get_model_serializers(AEDEncoder, model.model_config),
        label_scorer_configs=_get_label_scorer_configs(model=model, variant=variant),
        bpe_size=vocab_to_bpe_size(model.model_config.label_target_size),
        blank_index=blank_index,
        sentence_end_index=0,
        variant=variant,
        corpora=corpora,
    )


def run(
    model: TrainedAEDModel,
    variants: Optional[List[AEDRecogVariant]] = None,
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
