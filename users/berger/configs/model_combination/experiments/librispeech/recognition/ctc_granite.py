"""
Recognition with the IBM Granite Speech CTC model, an externally trained CTC model from HuggingFace.

Differences to the BPE CTC models of this setup that shape the search conditions:

* The label inventory is the model's own 16k subword vocabulary with the CTC blank at index 0, so the
  corpus BPE lexicon and the BPE level neural LMs do not apply. Word level LMs do, since they operate
  on lemma orthographies.
* The encoder frame shift is 80 ms (10 ms mel frames, stacked by 2, subsampled by 4) instead of 40 ms,
  which roughly halves the number of search steps.
* The model was trained on true-cased, punctuated text, so hypotheses are normalized before scoring.

Search parameters are taken over from the BPE CTC setup as a starting point; they are not tuned for
this model and the LM scales in particular should be swept.
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
from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
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
# Pin the first Transformers-format model snapshot (including processor/tokenizer files).
GRANITE_CTC_REVISION = "1ea89b7"
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
    normalize_lm_tokens: bool = False
    normalized_lexicon_mode: str = "canonical"


def run(
    model_descriptor: str = "ctc_granite-speech-5.0-470m-turboctc",
    huggingface_repo_dir: Optional[tk.Path] = None,
    variants: Optional[List[GraniteCTCRecogVariant]] = None,
    corpora: Optional[List[librispeech_datasets.EvalSet]] = None,
    prior_data_config: Optional[DataConfig] = None,
) -> List[RecogResult]:
    if huggingface_repo_dir is None:
        huggingface_repo_dir = get_huggingface_repo_dir(GRANITE_CTC_REPO_ID, revision=GRANITE_CTC_REVISION)

    if variants is None:
        variants = default_recog_variants()

    if corpora is None:
        corpora = librispeech_datasets.EVAL_SETS

    if prior_data_config is None:
        prior_data_config = librispeech_datasets.get_default_prior_data()

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
        default_offline_tree_trafo_recog_variant(),
        default_streaming_lexfree_recog_variant(),
        default_streaming_tree_4gram_recog_variant(),
    ]


def lm_scale_tuning_variants(
    lm_scales: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1),
) -> List[GraniteCTCRecogVariant]:
    """Return word-LM scale sweeps for the offline Granite tree searches."""

    variants = []
    for factory in (default_offline_tree_4gram_recog_variant, default_offline_tree_trafo_recog_variant):
        for lm_scale in lm_scales:
            variant = factory()
            assert variant.search_algorithm_params.word_lm_params is not None
            variant.search_algorithm_params.word_lm_params = replace(
                variant.search_algorithm_params.word_lm_params,
                scale=lm_scale,
            )
            variant.descriptor += f"_lm-scale-{lm_scale:g}"
            variants.append(variant)
    return variants


def text_normalized_lm_variants(
    lm_scales: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8),
) -> List[GraniteCTCRecogVariant]:
    """Published vs matched re-estimated original/HF-normalized 4-grams.

    All arms share prior scale, pruning, and HF scoring. Normalized arms compare
    canonical word spellings against recovered singleton/attached-suffix paths.
    Every decoding lemma has at most one LM token. LM scale zero measures the
    effect of changed acoustic paths without an LM-score contribution.
    The original-text re-estimation controls for corpus/estimator differences
    from the published OpenSLR LM. No Transformer training is scheduled.
    """
    from ....data.librispeech.lm_text_normalization import get_text_normalization_4gram_lms

    lms = get_text_normalization_4gram_lms()
    conditions = [
        ("published", None, None),
        ("original", lms["original"], None),
        ("normalized-canonical", lms["normalized"], "canonical"),
        ("normalized-split", lms["normalized"], "split"),
    ]
    variants = []
    for condition, lm_file, lexicon_mode in conditions:
        for lm_scale in lm_scales:
            variant = default_offline_tree_4gram_recog_variant()
            variant.descriptor = f"tree_4gram_textnorm-{condition}_lm-scale-{lm_scale:g}"
            variant.search_algorithm_params.word_lm_params = librispeech_lm.ArpaLmParams(
                scale=lm_scale, lm_file=lm_file
            )
            variant.normalize_lm_tokens = lexicon_mode is not None
            variant.normalized_lexicon_mode = lexicon_mode or "canonical"
            variants.append(variant)
    return variants


def default_offline_lexfree_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
    )


def default_offline_tree_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="tree",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[6.0],
            max_beam_sizes=[6],
            word_end_score_threshold=0.0,
            max_word_end_beam_size=1,
        ),
    )


def default_offline_tree_4gram_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
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


def default_offline_tree_trafo_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="tree_trafoLM",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[16.0],
            max_beam_sizes=[256],
            word_end_score_threshold=0.5,
            max_word_end_beam_size=16,
            word_lm_params=librispeech_lm.TransformerLmParams(scale=0.8),
        ),
        prior_scale=0.2,
    )


def default_streaming_lexfree_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="streaming_lexfree",
        search_algorithm_params=LexiconfreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[0.0],
            max_beam_sizes=[1],
        ),
        search_mode_params=_streaming_params(),
    )


def default_streaming_tree_4gram_recog_variant() -> GraniteCTCRecogVariant:
    return GraniteCTCRecogVariant(
        descriptor="streaming_tree_4gram",
        search_algorithm_params=LibrispeechTreeTimesyncRecogParams(
            collapse_repeated_labels=True,
            score_thresholds=[14.0],
            max_beam_sizes=[1024],
            word_end_score_threshold=0.5,
            word_lm_params=librispeech_lm.ArpaLmParams(scale=0.6),
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
    corpora: List[librispeech_datasets.EvalSet],
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
        normalize_lm_tokens=variant.normalize_lm_tokens,
        normalized_lexicon_mode=variant.normalized_lexicon_mode,
        recog_data_config_fn=lambda corpus: (
            librispeech_datasets.get_hf_recog_data(corpus)
            if corpus.startswith("test-")
            else librispeech_datasets.get_default_recog_data(corpus)
        ),
    )
