"""
Recognition with the IBM Granite Speech CTC model, an externally trained CTC model from HuggingFace.

See ``experiments/librispeech/recognition/ctc_granite.py`` for the properties of the model that shape
these search conditions. Note that the model is not trained on Loquacious, so this measures an
out-of-domain condition, while the word level LM and the lexicon remain corpus specific.
"""

__all__ = [
    "GraniteCTCRecogVariant",
    "default_recog_variants",
    "run",
]

from dataclasses import dataclass, field, fields, replace
from typing import List, Optional, Union

from i6_experiments.common.setups.serialization import Collection
from sisyphus import tk

from ....data.base import DataConfig
from ....data.huggingface import get_huggingface_repo_dir
from ....data.loquacious import datasets as loquacious_datasets
from ....data.loquacious import lm as loquacious_lm
from ....data.loquacious.recog import LoquaciousTreeTimesyncRecogParams
from ....model_pipelines.common.label_scorer_config import get_no_op_label_scorer_config
from ....model_pipelines.common.recog import OfflineRecogParameters, RecogResult, StreamingRecogParameters
from ....model_pipelines.common.recog_rasr_config import LexiconfreeTimesyncRecogParams
from ....model_pipelines.common.serializers import get_model_serializers
from ....model_pipelines.granite_ctc.prior import compute_priors
from ....model_pipelines.granite_ctc.pytorch_modules import (
    GraniteCtcConfig,
    GraniteCtcRecogConfig,
    GraniteCtcRecogModel,
)
from .common import BaseRecogVariant, run_single_hf_bpe_variant

GRANITE_CTC_REPO_ID = "ibm-granite/granite-speech-5.0-470m-turboctc-nc"
GRANITE_CTC_BLANK_INDEX = 0
GRANITE_CTC_FRAME_SHIFT_SECONDS = 0.08


def _offline_params(mem_rqmt: int = 24, gpu_mem_rqmt: int = 11) -> OfflineRecogParameters:
    # The 470M encoder is run on GPU; the search itself stays on the CPU.
    return OfflineRecogParameters(
        mem_rqmt=mem_rqmt,
        gpu_mem_rqmt=gpu_mem_rqmt,
        encoder_frame_shift_seconds=GRANITE_CTC_FRAME_SHIFT_SECONDS,
    )


def _streaming_params(mem_rqmt: int = 24, gpu_mem_rqmt: int = 11) -> StreamingRecogParameters:
    return StreamingRecogParameters(
        encoder_frame_shift_seconds=GRANITE_CTC_FRAME_SHIFT_SECONDS,
        mem_rqmt=mem_rqmt,
        gpu_mem_rqmt=gpu_mem_rqmt,
    )


@dataclass
class GraniteCTCRecogVariant(BaseRecogVariant):
    # Unlike the setup's own models, the encoder runs on GPU by default, see `_offline_params`.
    search_mode_params: Union[OfflineRecogParameters, StreamingRecogParameters] = field(
        default_factory=_offline_params
    )
    prior_scale: float = 0.0
    blank_penalty: float = 0.0


def run(
    model_descriptor: str = "ctc_granite-speech-5.0-470m-turboctc",
    huggingface_repo_dir: Optional[tk.Path] = None,
    variants: Optional[List[GraniteCTCRecogVariant]] = None,
    corpora: Optional[List[loquacious_datasets.EvalSet]] = None,
    prior_data_config: Optional[DataConfig] = None,
    prior_train_corpus_key: loquacious_datasets.TrainSet = "train.medium",
) -> List[RecogResult]:
    if huggingface_repo_dir is None:
        huggingface_repo_dir = get_huggingface_repo_dir(GRANITE_CTC_REPO_ID)

    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = loquacious_datasets.EVAL_SETS

    if prior_data_config is None:
        prior_data_config = loquacious_datasets.get_prior_data(prior_train_corpus_key)

    results = []

    for variant in variants:
        results.extend(
            _run_single_variant(
                model_descriptor=model_descriptor,
                huggingface_repo_dir=huggingface_repo_dir,
                variant=variant,
                corpora=corpora,
                prior_data_config=prior_data_config,
            )
        )
    return results


def default_recog_variants() -> List[GraniteCTCRecogVariant]:
    return [
        default_offline_lexfree_recog_variant(),
        default_offline_tree_recog_variant(),
        default_offline_tree_4gram_recog_variant(),
        default_streaming_lexfree_recog_variant(),
        default_streaming_tree_4gram_recog_variant(),
    ]


def lm_scale_tuning_variants(
    lm_scales: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5),
) -> List[GraniteCTCRecogVariant]:
    """Return word-4-gram LM scale sweeps for Granite on Loquacious."""

    variants = []
    for lm_scale in lm_scales:
        variant = default_offline_tree_4gram_recog_variant()
        assert variant.search_algorithm_params.word_lm_params is not None
        variant.search_algorithm_params.word_lm_params = replace(
            variant.search_algorithm_params.word_lm_params,
            scale=lm_scale,
        )
        variant.descriptor += f"_lm-scale-{lm_scale:g}"
        variants.append(variant)
    return variants


def default_offline_lexfree_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="recog_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
    )


def default_offline_tree_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="recog_tree",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[16],
            score_thresholds=[4.0],
        ),
    )


def default_offline_tree_4gram_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="recog_tree_4gram",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[64],
            score_thresholds=[6.0],
            word_lm_params=loquacious_lm.ArpaLmParams(scale=0.3),
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
        ),
        prior_scale=0.2,
    )


def default_streaming_lexfree_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="recog_streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1],
            score_thresholds=[0.0],
        ),
        search_mode_params=_streaming_params(),
    )


def default_streaming_tree_4gram_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="recog_streaming_tree_4gram",
        search_algorithm_params=LoquaciousTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            max_beam_sizes=[1024],
            score_thresholds=[14.0],
            word_end_score_threshold=0.5,
            word_lm_params=loquacious_lm.ArpaLmParams(scale=0.3),
        ),
        search_mode_params=_streaming_params(),
    )


def _get_model_serializers(
    huggingface_repo_dir: tk.Path,
    variant: GraniteCTCRecogVariant,
    prior_data_config: DataConfig,
) -> Collection:
    model_config = GraniteCtcConfig(
        huggingface_repo_dir=huggingface_repo_dir,
        blank_index=GRANITE_CTC_BLANK_INDEX,
    )
    if variant.prior_scale != 0.0:
        prior_file = compute_priors(prior_data_config=prior_data_config, model_config=model_config)
    return get_model_serializers(
        GraniteCtcRecogModel,
        GraniteCtcRecogConfig(
            **{f.name: getattr(model_config, f.name) for f in fields(model_config)},
            blank_penalty=variant.blank_penalty,
            prior_scale=variant.prior_scale,
            prior_file=prior_file if variant.prior_scale != 0.0 else None,
        ),
    )


def _run_single_variant(
    model_descriptor: str,
    huggingface_repo_dir: tk.Path,
    variant: GraniteCTCRecogVariant,
    corpora: List[loquacious_datasets.EvalSet],
    prior_data_config: DataConfig,
) -> List[RecogResult]:

    return run_single_hf_bpe_variant(
        model_descriptor=model_descriptor,
        huggingface_repo_dir=huggingface_repo_dir,
        encoder_serializers=_get_model_serializers(
            huggingface_repo_dir=huggingface_repo_dir, variant=variant, prior_data_config=prior_data_config
        ),
        label_scorer_configs=[get_no_op_label_scorer_config()],
        blank_index=GRANITE_CTC_BLANK_INDEX,
        variant=variant,
        corpora=corpora,
    )
