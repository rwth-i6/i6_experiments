__all__ = [
    "OOV_TARGET_RATES",
    "UNKNOWN_PROBABILITY_BOOSTS",
    "run",
    "setup_oov_lexicons",
    "setup_oov_lm_conditions",
]

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

from i6_core.corpus.stats import CountCorpusWordFrequenciesJob
from i6_core.lm.kenlm import KenLMplzJob
from i6_experiments.common.datasets.librispeech.corpus import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import get_librispeech_normalized_lm_data
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab
from sisyphus import tk

from ...data.librispeech import datasets as librispeech_datasets
from ...data.librispeech import lm as librispeech_lm
from ...data.librispeech.lexicon_oov import (
    CollapseOovScoresInOnnxJob,
    CountLexiconOovJob,
    LexiconToKenLmVocabularyJob,
    NormalizeArpaUnknownTokenJob,
    BoostArpaUnknownProbabilityJob,
    ReduceLexiconToTargetOovJob,
)
from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.train import TrainedModel
from ...tools import ken_lm_binaries
from .recognition import ctc_bpe, ffnn_transducer_bpe

OOV_TARGET_RATES: Tuple[float, ...] = (0.01, 0.015, 0.02, 0.03)
# Multiplicative probability factors for the unknown word in adapted 4-gram LMs.
# Values below one test whether the current unknown probability is too high.
UNKNOWN_PROBABILITY_BOOSTS: Tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
DEV_CORPORA: Tuple[librispeech_datasets.EvalSet, ...] = ("dev-other",)
OUTPUT_PREFIX = "datasets/LibriSpeech/lexicon_oov"

RecogVariant = TypeVar("RecogVariant", ctc_bpe.CTCRecogVariant, ffnn_transducer_bpe.TransducerRecogVariant)
VariantFactory = Callable[[], RecogVariant]


@dataclass(frozen=True)
class OovLexiconSetup:
    lexicon_file: tk.Path
    removed_lemmas: Optional[tk.Path]


@dataclass(frozen=True)
class OovLmCondition:
    lexicon_file: tk.Path
    adapted_arpa_lm: tk.Path
    adapted_transformer_onnx: tk.Path
    transformer_vocab: tk.Path
    boosted_arpa_lms: Dict[float, tk.Path]


def _rate_descriptor(rate: float) -> str:
    return f"{rate * 100:g}pct".replace(".", "p")


def _register_oov_job_outputs(job: CountLexiconOovJob, name: str) -> None:
    tk.register_output(f"{OUTPUT_PREFIX}/{name}/report", job.out_report)
    tk.register_output(f"{OUTPUT_PREFIX}/{name}/oov_words", job.out_oov_words)
    tk.register_output(f"{OUTPUT_PREFIX}/{name}/token_oov_rate", job.out_token_oov_rate)
    tk.register_output(f"{OUTPUT_PREFIX}/{name}/type_oov_rate", job.out_type_oov_rate)


@lru_cache
def _setup_oov_lexicons(target_oov_rates: Tuple[float, ...]) -> Dict[str, OovLexiconSetup]:
    """
    Create the standard and frequency-pruned word lexica used by the OOV comparison.

    Lemmas are ranked by their token counts in ``train-other-960``. Target rates are measured jointly on
    ``dev-clean`` and ``dev-other``; per-corpus OOV reports are generated as well.
    """

    corpus_files = get_bliss_corpus_dict("wav")
    dev_corpora = [corpus_files[corpus] for corpus in DEV_CORPORA]
    standard_lexicon = get_bliss_lexicon()
    tk.register_output(f"{OUTPUT_PREFIX}/standard/lexicon", standard_lexicon)

    frequency_job = CountCorpusWordFrequenciesJob(corpus_files["train-other-960"])
    frequency_job.add_alias(f"{OUTPUT_PREFIX}/training_word_frequencies")
    tk.register_output(f"{OUTPUT_PREFIX}/training_word_frequencies", frequency_job.out_word_counts)

    standard_combined_oov_job = CountLexiconOovJob(
        bliss_corpora=dev_corpora,
        bliss_lexicon=standard_lexicon,
    )
    standard_combined_oov_job.add_alias(f"{OUTPUT_PREFIX}/standard/dev-all")
    _register_oov_job_outputs(standard_combined_oov_job, "standard/dev-all")
    for corpus in DEV_CORPORA:
        oov_job = CountLexiconOovJob(
            bliss_corpora=[corpus_files[corpus]],
            bliss_lexicon=standard_lexicon,
        )
        oov_job.add_alias(f"{OUTPUT_PREFIX}/standard/{corpus}")
        _register_oov_job_outputs(oov_job, f"standard/{corpus}")

    lexicons = {"standard": OovLexiconSetup(lexicon_file=standard_lexicon, removed_lemmas=None)}
    for target_oov_rate in target_oov_rates:
        rate_descriptor = _rate_descriptor(target_oov_rate)
        reduction_job = ReduceLexiconToTargetOovJob(
            bliss_lexicon=standard_lexicon,
            evaluation_corpora=dev_corpora,
            word_frequencies=frequency_job.out_word_counts,
            target_oov_rate=target_oov_rate,
        )
        reduction_job.add_alias(f"{OUTPUT_PREFIX}/{rate_descriptor}/reduce_lexicon")
        tk.register_output(f"{OUTPUT_PREFIX}/{rate_descriptor}/lexicon", reduction_job.out_lexicon)
        tk.register_output(f"{OUTPUT_PREFIX}/{rate_descriptor}/report", reduction_job.out_report)
        tk.register_output(f"{OUTPUT_PREFIX}/{rate_descriptor}/removed_lemmas", reduction_job.out_removed_lemmas)
        tk.register_output(
            f"{OUTPUT_PREFIX}/{rate_descriptor}/token_oov_rate",
            reduction_job.out_token_oov_rate,
        )
        tk.register_output(
            f"{OUTPUT_PREFIX}/{rate_descriptor}/num_removed_lemmas",
            reduction_job.out_num_removed_lemmas,
        )
        lexicons[rate_descriptor] = OovLexiconSetup(
            lexicon_file=reduction_job.out_lexicon,
            removed_lemmas=reduction_job.out_removed_lemmas,
        )

        for corpus in DEV_CORPORA:
            oov_job = CountLexiconOovJob(
                bliss_corpora=[corpus_files[corpus]],
                bliss_lexicon=reduction_job.out_lexicon,
            )
            oov_job.add_alias(f"{OUTPUT_PREFIX}/{rate_descriptor}/{corpus}")
            _register_oov_job_outputs(oov_job, f"{rate_descriptor}/{corpus}")

    return lexicons


def setup_oov_lexicons(target_oov_rates: Sequence[float] = OOV_TARGET_RATES) -> Dict[str, tk.Path]:
    """Return the standard and reduced search lexica while registering their OOV statistics."""

    return {
        descriptor: setup.lexicon_file for descriptor, setup in _setup_oov_lexicons(tuple(target_oov_rates)).items()
    }


def setup_oov_lm_conditions(target_oov_rates: Sequence[float] = OOV_TARGET_RATES) -> Dict[str, OovLmCondition]:
    """Re-estimate the word 4-gram and adapt the word-Transformer output for every reduced lexicon."""

    lexicon_setups = _setup_oov_lexicons(tuple(target_oov_rates))
    lm_training_text = get_librispeech_normalized_lm_data()
    transformer_vocab = get_lm_vocab(output_prefix="").vocab
    transformer_params = librispeech_lm.TransformerLmParams()
    base_transformer_onnx = librispeech_lm.get_word_transformer_onnx_model(transformer_params)

    conditions = {}
    for descriptor, lexicon_setup in lexicon_setups.items():
        output_prefix = f"{OUTPUT_PREFIX}/{descriptor}/lm-adapted"
        vocabulary_job = LexiconToKenLmVocabularyJob(lexicon_setup.lexicon_file)
        vocabulary_job.add_alias(f"{output_prefix}/arpa/vocabulary")
        tk.register_output(f"{output_prefix}/arpa/vocabulary", vocabulary_job.out_vocabulary)

        # --limit_vocab_file maps every excluded training token to KenLM's <unk> before n-gram estimation.
        arpa_job = KenLMplzJob(
            text=[lm_training_text],
            order=4,
            interpolate_unigrams=False,
            pruning=[0, 0, 1, 1],
            vocabulary=vocabulary_job.out_vocabulary,
            discount_fallback=(),
            kenlm_binary_folder=ken_lm_binaries,
            mem=100,
            time=48,
        )
        arpa_job.add_alias(f"{output_prefix}/arpa/estimate_4gram")
        normalized_arpa_job = NormalizeArpaUnknownTokenJob(arpa_job.out_lm)
        normalized_arpa_job.add_alias(f"{output_prefix}/arpa/normalize_unknown")
        tk.register_output(f"{output_prefix}/arpa/4gram", normalized_arpa_job.out_arpa_lm)
        boosted_arpa_lms = {}
        for boost in UNKNOWN_PROBABILITY_BOOSTS:
            if boost == 1.0:
                boosted_arpa_lms[boost] = normalized_arpa_job.out_arpa_lm
                continue
            boost_job = BoostArpaUnknownProbabilityJob(normalized_arpa_job.out_arpa_lm, probability_factor=boost)
            boost_desc = f"unknown-probability-x{boost:g}".replace(".", "p")
            boost_job.add_alias(f"{output_prefix}/arpa/{boost_desc}")
            tk.register_output(f"{output_prefix}/arpa/{boost_desc}", boost_job.out_arpa_lm)
            boosted_arpa_lms[boost] = boost_job.out_arpa_lm

        if lexicon_setup.removed_lemmas is None:
            adapted_transformer_onnx = base_transformer_onnx
        else:
            onnx_adaptation_job = CollapseOovScoresInOnnxJob(
                onnx_model=base_transformer_onnx,
                lm_vocab=transformer_vocab,
                removed_lemmas=lexicon_setup.removed_lemmas,
            )
            onnx_adaptation_job.add_alias(f"{output_prefix}/word_transformer/collapse_oov_scores")
            tk.register_output(f"{output_prefix}/word_transformer/report", onnx_adaptation_job.out_report)
            tk.register_output(
                f"{output_prefix}/word_transformer/removed_indices",
                onnx_adaptation_job.out_removed_indices,
            )
            adapted_transformer_onnx = onnx_adaptation_job.out_onnx_model
        tk.register_output(f"{output_prefix}/word_transformer/model", adapted_transformer_onnx)

        conditions[descriptor] = OovLmCondition(
            lexicon_file=lexicon_setup.lexicon_file,
            adapted_arpa_lm=normalized_arpa_job.out_arpa_lm,
            adapted_transformer_onnx=adapted_transformer_onnx,
            transformer_vocab=transformer_vocab,
            boosted_arpa_lms=boosted_arpa_lms,
        )
    return conditions


def _make_variants(
    *,
    factories: Sequence[VariantFactory],
    lexicons: Optional[Dict[str, tk.Path]],
) -> List[RecogVariant]:
    variants = []
    if lexicons is None:
        for factory in factories:
            variant = factory()
            variant.descriptor = f"lexicon-oov_lexfree_{variant.descriptor}"
            variants.append(variant)
    else:
        for lexicon_descriptor, lexicon_file in lexicons.items():
            for factory in factories:
                # Keep the closed-vocabulary tree as a control and add a
                # paired open-vocabulary tree using RASR's OOV subtree.
                for use_oov_fallback in (False, True):
                    variant = factory()
                    variant.descriptor = f"lexicon-oov_{lexicon_descriptor}_{variant.descriptor}"
                    if use_oov_fallback:
                        variant.descriptor += "_oov"
                        variant.use_oov_fallback = True
                        variant.search_algorithm_params = replace(
                            variant.search_algorithm_params,
                            sentence_end_fallback=False,
                        )
                    variant.base_lexicon_file = lexicon_file
                    variants.append(variant)
    return variants


def _get_ctc_variants(lexicons: Dict[str, tk.Path]) -> List[ctc_bpe.CTCRecogVariant]:
    lexfree_factories = [
        ctc_bpe.default_offline_lexfree_recog_variant,
        ctc_bpe.default_offline_lexfree_lstm_recog_variant,
        ctc_bpe.default_offline_lexfree_trafo_recog_variant,
    ]
    tree_factories = [
        ctc_bpe.default_offline_tree_recog_variant,
        ctc_bpe.default_offline_tree_4gram_recog_variant,
        ctc_bpe.default_offline_tree_lstm_recog_variant,
        ctc_bpe.default_offline_tree_lstm_4gram_recog_variant,
        ctc_bpe.default_offline_tree_trafo_recog_variant,
    ]
    return _make_variants(factories=lexfree_factories, lexicons=None) + _make_variants(
        factories=tree_factories,
        lexicons=lexicons,
    )


def _make_lm_adapted_variants(
    *,
    arpa_factories: Sequence[VariantFactory],
    transformer_factory: VariantFactory,
    conditions: Dict[str, OovLmCondition],
) -> List[RecogVariant]:
    variants = []
    for condition_descriptor, condition in conditions.items():
        for factory in arpa_factories:
            for use_oov_fallback in (False, True):
                boosts = UNKNOWN_PROBABILITY_BOOSTS if use_oov_fallback else (1.0,)
                for boost in boosts:
                    variant = factory()
                    word_lm_params = variant.search_algorithm_params.word_lm_params
                    assert isinstance(word_lm_params, librispeech_lm.ArpaLmParams)
                    variant.base_lexicon_file = condition.lexicon_file
                    variant.search_algorithm_params.word_lm_params = replace(
                        word_lm_params, lm_file=condition.boosted_arpa_lms[boost]
                    )
                    variant.descriptor = f"lexicon-oov_{condition_descriptor}_lm-adapted_{variant.descriptor}"
                    if use_oov_fallback:
                        variant.descriptor += "_oov"
                        if boost != 1.0:
                            variant.descriptor += f"_unkboost{boost:g}".replace(".", "p")
                        variant.use_oov_fallback = True
                        variant.search_algorithm_params = replace(
                            variant.search_algorithm_params,
                            sentence_end_fallback=False,
                        )
                    variants.append(variant)

        # For the standard lexicon this would be byte-identical to the existing frozen Transformer condition.
        if condition_descriptor != "standard":
            for use_oov_fallback in (False, True):
                variant = transformer_factory()
                word_lm_params = variant.search_algorithm_params.word_lm_params
                assert isinstance(word_lm_params, librispeech_lm.TransformerLmParams)
                variant.search_algorithm_params.word_lm_params = replace(
                    word_lm_params,
                    onnx_model=condition.adapted_transformer_onnx,
                    vocab_file=condition.transformer_vocab,
                )
                variant.base_lexicon_file = condition.lexicon_file
                variant.descriptor = f"lexicon-oov_{condition_descriptor}_lm-adapted_{variant.descriptor}"
                if use_oov_fallback:
                    variant.descriptor += "_oov"
                    variant.use_oov_fallback = True
                    variant.search_algorithm_params = replace(
                        variant.search_algorithm_params,
                        sentence_end_fallback=False,
                    )
                variants.append(variant)
    return variants


def _get_transducer_variants(
    lexicons: Dict[str, tk.Path],
) -> List[ffnn_transducer_bpe.TransducerRecogVariant]:
    lexfree_factories = [
        ffnn_transducer_bpe.default_offline_lexfree_recog_variant,
        ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant,
        ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant,
    ]
    tree_factories = [
        ffnn_transducer_bpe.default_offline_tree_recog_variant,
        ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant,
        ffnn_transducer_bpe.default_offline_tree_lstm_recog_variant,
        ffnn_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant,
        ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant,
    ]
    return _make_variants(factories=lexfree_factories, lexicons=None) + _make_variants(
        factories=tree_factories,
        lexicons=lexicons,
    )


def run(
    ctc_model: TrainedModel,
    transducer_model: TrainedModel,
    target_oov_rates: Sequence[float] = OOV_TARGET_RATES,
) -> List[RecogResult]:
    """
    Compare lexicon-free and tree search for CTC and transducer models.

    The original-LM variants isolate the effect of tree pruning. Additional LM-adapted variants re-estimate the ARPA
    LM with excluded words mapped to unknown and collapse the corresponding word-Transformer outputs into unknown.
    BPE neural LMs remain unchanged because removed words do not correspond to individual BPE output labels.
    """

    lm_conditions = setup_oov_lm_conditions(target_oov_rates=target_oov_rates)
    lexicons = {descriptor: condition.lexicon_file for descriptor, condition in lm_conditions.items()}
    ctc_variants = _get_ctc_variants(lexicons)
    ctc_variants.extend(
        _make_lm_adapted_variants(
            arpa_factories=[
                ctc_bpe.default_offline_tree_4gram_recog_variant,
                ctc_bpe.default_offline_tree_lstm_4gram_recog_variant,
            ],
            transformer_factory=ctc_bpe.default_offline_tree_trafo_recog_variant,
            conditions=lm_conditions,
        )
    )
    results = ctc_bpe.run(
        model=ctc_model,
        variants=ctc_variants,
        corpora=list(DEV_CORPORA),
    )
    transducer_variants = _get_transducer_variants(lexicons)
    transducer_variants.extend(
        _make_lm_adapted_variants(
            arpa_factories=[
                ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant,
                ffnn_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant,
            ],
            transformer_factory=ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant,
            conditions=lm_conditions,
        )
    )
    results.extend(
        ffnn_transducer_bpe.run(
            model=transducer_model,
            variants=transducer_variants,
            corpora=list(DEV_CORPORA),
        )
    )
    return results
