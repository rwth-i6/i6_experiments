__all__ = [
    "BaseRecogVariant",
    "run_single_bpe_variant",
    "run_single_byte_variant",
    "run_single_hf_bpe_variant",
    "run_single_phoneme_variant",
]

from dataclasses import dataclass, field, replace
from typing import Any, List, Optional, Sequence, Union

from i6_core.rasr import RasrConfig
from i6_core.returnn import PtCheckpoint
from i6_core.util import DelayedFormat
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.setups.serialization import Call, Collection, Import
from sisyphus import tk

from ....data.base import get_byte_vocab_file
from ....data.huggingface import DEFAULT_CASINGS, get_hf_bpe_bliss_lexicon, get_hf_ctc_vocab_file
from ....data.librispeech import datasets as librispeech_datasets
from ....data.librispeech import lm as librispeech_lm
from ....data.librispeech.bpe import get_bpe_vocab_file
from ....data.librispeech.lexicon import get_bliss_phoneme_lexicon, get_bpe_bliss_lexicon
from ....data.librispeech.recog import LibrispeechTreeTimesyncRecogParams
from ....model_pipelines.common.recog import (
    HuggingFaceTracebackFormatter,
    OfflineRecogParameters,
    RecogResult,
    StreamingRecogParameters,
    Utf8ByteTracebackFormatter,
    recog_rasr_offline,
    recog_rasr_streaming,
)
from ....model_pipelines.common.recog_rasr_config import (
    LexiconfreeLabelsyncRecogParams,
    LexiconfreeTimesyncRecogParams,
    get_lexiconfree_labelsync_recog_config,
    get_lexiconfree_timesync_recog_config,
    get_tree_timesync_recog_config,
)


@dataclass
class BaseRecogVariant:
    descriptor: str
    search_algorithm_params: Union[
        LexiconfreeLabelsyncRecogParams, LexiconfreeTimesyncRecogParams, LibrispeechTreeTimesyncRecogParams
    ]
    search_mode_params: Union[OfflineRecogParameters, StreamingRecogParameters] = field(
        default_factory=OfflineRecogParameters
    )
    compute_search_errors: bool = False
    base_lexicon_file: Optional[tk.Path] = None
    use_oov_fallback: bool = False
    normalize_hf_score: bool = False
    # Optional prototype overrides; existing variants keep their old job inputs.
    rasr_binary_path_override: Optional[tk.Path] = None
    prepared_lexicon_file: Optional[tk.Path] = None
    prototype_traceback_formatter_serializers: Optional[List[Any]] = None


def run_single_bpe_variant(
    model_descriptor: str,
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    bpe_size: int,
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    rasr_init_hook: Optional[Any] = None,
) -> List[RecogResult]:
    use_blank = blank_index is not None
    use_sentence_end = sentence_end_index is not None
    vocab_file = get_bpe_vocab_file(bpe_size=bpe_size, add_blank=use_blank)
    lexicon_file = variant.prepared_lexicon_file or get_bpe_bliss_lexicon(
        bpe_size=bpe_size,
        add_blank=use_blank,
        add_sentence_end_pron=use_sentence_end,
        base_lexicon_path=variant.base_lexicon_file,
        add_oov_fallback=variant.use_oov_fallback,
    )

    return _run_single_variant(
        model_descriptor=model_descriptor,
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=label_scorer_configs,
        vocab_file=vocab_file,
        lexicon_file=lexicon_file,
        blank_index=blank_index,
        sentence_end_index=sentence_end_index,
        variant=variant,
        corpora=corpora,
        rasr_init_hook=rasr_init_hook,
        traceback_formatter_serializers=variant.prototype_traceback_formatter_serializers,
    )


def run_single_byte_variant(
    model_descriptor: str,
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
) -> List[RecogResult]:
    use_blank = blank_index is not None
    vocab_file = get_byte_vocab_file(add_blank=use_blank)
    traceback_formatter_serializers = [
        Import(f"{Utf8ByteTracebackFormatter.__module__}.{Utf8ByteTracebackFormatter.__name__}"),
        Call(Utf8ByteTracebackFormatter.__name__, return_assign_variables="traceback_formatter"),
    ]

    return _run_single_variant(
        model_descriptor=model_descriptor,
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=label_scorer_configs,
        vocab_file=vocab_file,
        lexicon_file=None,
        blank_index=blank_index,
        sentence_end_index=sentence_end_index,
        variant=variant,
        corpora=corpora,
        traceback_formatter_serializers=traceback_formatter_serializers,
    )


def run_single_hf_bpe_variant(
    model_descriptor: str,
    huggingface_repo_dir: tk.Path,
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    blank_index: int,
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    checkpoint: Optional[PtCheckpoint] = None,
    casings: Sequence[str] = DEFAULT_CASINGS,
    recog_data_config_fn: Optional[Any] = None,
    normalize_lm_tokens: bool = False,
    normalized_lexicon_mode: str = "canonical",
) -> List[RecogResult]:
    """
    Recognition with an externally trained model that brings its own subword inventory.

    The label inventory is derived from the model's tokenizer instead of the corpus BPE codes. The
    lemma orthographies normally stay those of the corpus lexicon. With
    normalize_lm_tokens=True, use a normalized-word search lexicon instead;
    it has zero or one syntactic token per lemma, never multi-token exits.
    """
    variant.normalize_hf_score = True
    vocab_file = get_hf_ctc_vocab_file(huggingface_repo_dir=huggingface_repo_dir, blank_index=blank_index)
    lexicon_file = variant.prepared_lexicon_file or get_hf_bpe_bliss_lexicon(
        bliss_lexicon=variant.base_lexicon_file if variant.base_lexicon_file is not None else get_bliss_lexicon(),
        huggingface_repo_dir=huggingface_repo_dir,
        blank_index=blank_index,
        casings=casings,
    )
    if normalize_lm_tokens:
        from ....data.librispeech.lm_text_normalization import (
            NormalizeLexiconLmTokensJob,
            BuildNormalizedWordSearchLexiconJob,
        )

        if not isinstance(variant.search_algorithm_params, LibrispeechTreeTimesyncRecogParams):
            raise ValueError("Normalized word-LM tokens require tree search")
        normalized = NormalizeLexiconLmTokensJob(lexicon_file)
        search_lexicon = BuildNormalizedWordSearchLexiconJob(
            normalized.out_lexicon, huggingface_repo_dir, mode=normalized_lexicon_mode, casings=casings
        )
        prefix = f"recognition/{model_descriptor}__recog_{variant.descriptor}/normalized_lexicon"
        search_lexicon.add_alias(prefix)
        tk.register_output(f"{prefix}/lexicon", search_lexicon.out_lexicon)
        tk.register_output(f"{prefix}/report", search_lexicon.out_report)
        tk.register_output(f"{prefix}/uncovered_pronunciations", search_lexicon.out_uncovered)
        lexicon_file = search_lexicon.out_lexicon

    if isinstance(variant.search_algorithm_params, LibrispeechTreeTimesyncRecogParams):
        # Tree search emits word lemmas (original or normalized orthographies).
        traceback_formatter_serializers = variant.prototype_traceback_formatter_serializers
    else:
        # Lexicon-free search emits tokenizer pieces, which have to be decoded back into words. The
        # model was trained on true-cased, punctuated text while the references are uppercase without
        # punctuation, so the decoded text is normalized before scoring.
        traceback_formatter_serializers = [
            Import(f"{HuggingFaceTracebackFormatter.__module__}.{HuggingFaceTracebackFormatter.__name__}"),
            Call(
                HuggingFaceTracebackFormatter.__name__,
                kwargs=[
                    ("huggingface_repo_dir", DelayedFormat('tk.Path("{}")', huggingface_repo_dir)),
                    ("case_normalization", repr("upper")),
                    ("remove_punctuation", repr(True)),
                ],
                return_assign_variables="traceback_formatter",
            ),
        ]

    return _run_single_variant(
        model_descriptor=model_descriptor,
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=label_scorer_configs,
        vocab_file=vocab_file,
        lexicon_file=lexicon_file,
        blank_index=blank_index,
        sentence_end_index=None,
        variant=variant,
        corpora=corpora,
        traceback_formatter_serializers=traceback_formatter_serializers,
        recog_data_config_fn=recog_data_config_fn,
    )


def run_single_phoneme_variant(
    model_descriptor: str,
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    traceback_formatter_serializers: Optional[List[Any]] = None,
) -> List[RecogResult]:
    assert not isinstance(variant.search_algorithm_params, LexiconfreeTimesyncRecogParams)
    assert not isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams)
    lexicon_file = get_bliss_phoneme_lexicon()

    return _run_single_variant(
        model_descriptor=model_descriptor,
        checkpoint=checkpoint,
        encoder_serializers=encoder_serializers,
        label_scorer_configs=label_scorer_configs,
        vocab_file=None,
        lexicon_file=lexicon_file,
        blank_index=blank_index,
        sentence_end_index=sentence_end_index,
        variant=variant,
        corpora=corpora,
        traceback_formatter_serializers=traceback_formatter_serializers,
    )


def _run_single_variant(
    model_descriptor: str,
    checkpoint: Optional[PtCheckpoint],
    encoder_serializers: Collection,
    label_scorer_configs: List[RasrConfig],
    vocab_file: Optional[tk.Path],
    lexicon_file: Optional[tk.Path],
    blank_index: Optional[int],
    sentence_end_index: Optional[int],
    variant: BaseRecogVariant,
    corpora: List[librispeech_datasets.EvalSet],
    traceback_formatter_serializers: Optional[List[Any]] = None,
    rasr_init_hook: Optional[Any] = None,
    recog_data_config_fn: Optional[Any] = None,
) -> List[RecogResult]:
    if variant.rasr_binary_path_override is not None and not isinstance(
        variant.search_mode_params, OfflineRecogParameters
    ):
        raise NotImplementedError("Alternate RASR builds are currently wired only for offline recognition")
    if isinstance(variant.search_algorithm_params, LexiconfreeLabelsyncRecogParams):
        assert vocab_file is not None
        if variant.compute_search_errors:
            raise NotImplementedError
        recog_config = get_lexiconfree_labelsync_recog_config(
            vocab_file=vocab_file,
            label_scorer_configs=label_scorer_configs,
            params=variant.search_algorithm_params,
            sentence_end_index=sentence_end_index,
            logfile_suffix="recog",
        )
        align_config = None
    elif isinstance(variant.search_algorithm_params, LexiconfreeTimesyncRecogParams):
        assert vocab_file is not None
        if variant.compute_search_errors:
            raise NotImplementedError
        recog_config = get_lexiconfree_timesync_recog_config(
            vocab_file=vocab_file,
            label_scorer_configs=label_scorer_configs,
            params=variant.search_algorithm_params,
            sentence_end_index=sentence_end_index,
            blank_index=blank_index,
            logfile_suffix="recog",
        )
        align_config = None
    elif isinstance(variant.search_algorithm_params, LibrispeechTreeTimesyncRecogParams):
        assert lexicon_file is not None
        if variant.search_algorithm_params.word_lm_params is not None:
            lm_config = librispeech_lm.get_word_lm_config(
                lexicon_file=lexicon_file,
                params=variant.search_algorithm_params.word_lm_params,
                **(
                    {"rasr_binary_path_override": variant.rasr_binary_path_override}
                    if variant.rasr_binary_path_override is not None
                    else {}
                ),
            )
        else:
            lm_config = None
        recog_config = get_tree_timesync_recog_config(
            lexicon_file=lexicon_file,
            label_scorer_configs=label_scorer_configs,
            params=variant.search_algorithm_params,
            lm_config=lm_config,
            logfile_suffix="recog",
        )
        if variant.compute_search_errors:
            align_params = replace(
                variant.search_algorithm_params,
                max_beam_sizes=[2048] * len(variant.search_algorithm_params.max_beam_sizes),
                score_thresholds=[22.0] * len(variant.search_algorithm_params.max_beam_sizes),
                max_word_end_beam_size=None,
                word_end_score_threshold=None,
                maximum_stable_delay=None,
                maximum_stable_delay_pruning_interval=None,
            )
            align_config = get_tree_timesync_recog_config(
                lexicon_file=lexicon_file,
                label_scorer_configs=label_scorer_configs,
                params=align_params,
                lm_config=lm_config,
                logfile_suffix="align",
            )
        else:
            align_config = None

    results = []
    for corpus in corpora:
        recog_data = recog_data_config_fn(corpus) if recog_data_config_fn is not None else librispeech_datasets.get_default_recog_data(corpus)
        if variant.normalize_hf_score and hasattr(recog_data, "peak_normalization"):
            # HF models receive the processor's raw waveform: no RASR-style
            # peak normalization or pre-emphasis.
            recog_data = replace(recog_data, peak_normalization=False, preemphasis=None)
        score_corpus = librispeech_datasets.get_default_score_corpus(corpus)

        if isinstance(variant.search_mode_params, OfflineRecogParameters):
            recog_result = recog_rasr_offline(
                descriptor=f"{model_descriptor}__recog_{variant.descriptor}",
                checkpoint=checkpoint,
                recog_rasr_config_file=recog_config,
                align_rasr_config_file=align_config,
                recog_data_config=recog_data,
                recog_corpus=score_corpus,
                encoder_serializers=encoder_serializers,
                sample_rate=16000,
                params=variant.search_mode_params,
                traceback_formatter_serializers=traceback_formatter_serializers,
                rasr_init_hook=rasr_init_hook,
                normalize_hf_score=variant.normalize_hf_score,
                **(
                    {"rasr_binary_path_override": variant.rasr_binary_path_override}
                    if variant.rasr_binary_path_override is not None
                    else {}
                ),
            )
        elif isinstance(variant.search_mode_params, StreamingRecogParameters):
            recog_result = recog_rasr_streaming(
                descriptor=f"{model_descriptor}__recog_{variant.descriptor}",
                checkpoint=checkpoint,
                recog_rasr_config_file=recog_config,
                recog_data_config=recog_data,
                recog_corpus=score_corpus,
                encoder_serializers=encoder_serializers,
                sample_rate=16000,
                params=variant.search_mode_params,
                traceback_formatter_serializers=traceback_formatter_serializers,
                rasr_init_hook=rasr_init_hook,
                normalize_hf_score=variant.normalize_hf_score,
            )

        results.append(recog_result)

    return results
