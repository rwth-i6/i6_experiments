"""Full-lexicon CTC OOV prototypes on dev-other, pinned to RASR 5083edde.

Closed baselines are rerun on the same master-based binary (no LM look-ahead).
No reduced lexica, adapted LMs, neural LMs, or held-out tuning are mixed in here.
"""

from dataclasses import replace

from i6_core.am.config import TdpValues, acoustic_model_config
from i6_core.lexicon.allophones import DumpStateTyingJob
from i6_core.rasr import CommonRasrParameters, RasrConfig, crp_add_default_output
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_experiments.common.setups.serialization import Call, Import
from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat

from ... import tools
from ...data.huggingface import HuggingFaceBpeBlissLexiconJob, get_hf_ctc_vocab_file, get_huggingface_repo_dir
from ...data.librispeech import datasets
from ...data.librispeech.bpe import get_bpe_vocab_file, vocab_to_bpe_size
from ...data.librispeech.lexicon import get_bpe_bliss_lexicon
from ...data.librispeech.lm import ArpaLmParams
from ...data.oov_decoder_smoke import OovPrototypeDecoderSmokeJob
from ...data.oov_prototype import (
    BuildGraniteOovLexiconJob,
    InspectWordStartTokenizerJob,
    RemoveEmptyUnknownPronunciationsJob,
    ValidateOovPrototypeInputsJob,
)
from ...model_pipelines.common.oov_traceback import GraniteOovTracebackFormatter
from .recognition import ctc_bpe, ctc_granite


PENALTIES = (-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0)
BPE_POSITIVE_PENALTIES = (6.0, 8.0, 10.0, 12.0)
BPE_CLOSED_LIMIT_PENALTIES = tuple(float(beta) for beta in range(10, 101, 10))


def prototype_variants(kind):
    """Controlled beta x LM-scale grid, plus matched beam convergence probes.

    All tree arms use prior=0, blank penalty=0, Viterbi recombination, no word
    reward and strict sentence-end handling. LM-free really has no word LM.
    """
    if kind not in {"bpe", "granite"}:
        raise ValueError(kind)
    module = ctc_bpe if kind == "bpe" else ctc_granite
    tokenization = "continuation-marked" if kind == "bpe" else "word-start-marked"
    lm_scales = (0.2, 0.4, 0.6) if kind == "bpe" else (0.1, 0.3, 0.5)
    variants = []
    seen = set()

    def add(mode, scale, beta=0.0, beam=256, threshold=16.0):
        key = (mode, scale, beta, beam, threshold)
        if key in seen:
            return
        seen.add(key)
        variant = module.default_offline_tree_recog_variant()
        variant.prior_scale = 0.0
        variant.blank_penalty = 0.0
        lm_name = "no-lm" if scale is None else f"4gram-lm{scale:g}"
        variant.descriptor = f"oov-prototype-5083edde_{mode}_{lm_name}_beta{beta:g}_beam{beam}_thr{threshold:g}"
        variant.search_algorithm_params = replace(
            variant.search_algorithm_params,
            word_lm_params=None if scale is None else ArpaLmParams(scale=scale),
            max_beam_sizes=[beam],
            score_thresholds=[threshold],
            max_word_end_beam_size=beam,
            word_end_score_threshold=threshold,
            sentence_end_fallback=False,
            recombination_mode="on",
            unknown_word_fallback=mode,
            unknown_word_tokenization=tokenization,
            unknown_word_penalty=beta,
        )
        variant.search_mode_params = replace(variant.search_mode_params, mem_rqmt=32)
        variants.append(variant)

    for scale in (None, *lm_scales):
        add("disabled", scale)
        if kind == "bpe":
            add("legacy", scale)
        for beta in PENALTIES:
            add("known-excluding", scale, beta)
    # Same-pruning A/B at wider beams, distinct from penalty/LM tuning.
    for scale in (None, lm_scales[1]):
        for mode in ("disabled", "known-excluding"):
            for beam in (64, 1024):
                add(mode, scale, beam=beam)
    # Wide finite-beam topology smoke condition: not an exhaustive-search proof.
    for mode in ("disabled", "known-excluding"):
        add(mode, None, beam=4096, threshold=30.0)
    greedy = module.default_offline_lexfree_recog_variant()
    greedy.descriptor = "oov-prototype-5083edde_lexfree-greedy"
    variants.append(greedy)
    if kind == "bpe":
        # Follow-up to the first dev-other sweep: stronger LM scales encourage
        # invented/merged words at beta <= 4. Probe stronger UNK costs at fixed
        # pruning, retaining the original grid and its same-scale closed controls.
        # Append after the old conditions so their local ordering stays unchanged.
        for scale in (0.4, 0.6):
            for beta in BPE_POSITIVE_PENALTIES:
                add("known-excluding", scale, beta)
        # Hold the closed-vocabulary-tuned LM scale fixed while testing whether
        # suppressing unknown completions recovers the matched closed baseline.
        # add() reuses the existing beta=10 condition instead of duplicating it.
        for beta in BPE_CLOSED_LIMIT_PENALTIES:
            add("known-excluding", 0.6, beta)
    return variants


def _validated_lexicon(
    lexicon, vocab, binary, prefix, label_aliases=None, decoder_smoke_report=None, pruning_report=None, arpa_lm=None
):
    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.set_executables(rasr_binary_path=binary)
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon
    crp.acoustic_model_config = acoustic_model_config(
        state_tying="monophone",
        states_per_phone=1,
        phon_history_length=0,
        phon_future_length=0,
        tdp_transition=TdpValues(0.0, 0.0, "infinity", 0.0),
        tdp_silence=TdpValues(0.0, 0.0, "infinity", 0.0),
    )
    crp.acoustic_model_config.allophones.add_from_lexicon = False
    # Enumerate in RASR: its text allophone loader discards lines beginning
    # with '#', including the real Granite '#' phoneme. With zero context,
    # add-all generates exactly the four boundary variants of every phoneme.
    crp.acoustic_model_config.allophones.add_all = True
    dump = DumpStateTyingJob(crp)
    dump.add_alias(f"{prefix}/dump-state-tying")
    checked = ValidateOovPrototypeInputsJob(
        lexicon,
        vocab,
        dump.out_state_tying,
        arpa_lm if arpa_lm is not None else get_arpa_lm_dict()["4gram"],
        label_aliases=label_aliases,
        decoder_smoke_report=decoder_smoke_report,
        pruning_report=pruning_report,
    )
    checked.add_alias(f"{prefix}/validate")
    tk.register_output(f"{prefix}/state-tying", dump.out_state_tying)
    tk.register_output(f"{prefix}/validation", checked.out_report)
    tk.register_output(f"{prefix}/lexicon", checked.out_lexicon)
    return checked.out_lexicon


def run(ctc_model):
    """Register only dev-other exploration, reusing the existing BPE checkpoint."""
    binary = tools.get_oov_prototype_rasr_binary_path()
    prefix = "datasets/LibriSpeech/oov-prototype-5083edde"
    smoke = OovPrototypeDecoderSmokeJob(binary, tools.returnn_python_exe)
    smoke.add_alias("tools/rasr-oov-prototype/decoder-smoke")
    tk.register_output("tools/rasr-oov-prototype/decoder-smoke-report", smoke.out_report)
    tk.register_output("tools/rasr-oov-prototype/decoder-smoke-logs", smoke.out_logs)
    bpe_size = vocab_to_bpe_size(ctc_model.model_config.target_size - 1)
    bpe_lexicon = _validated_lexicon(
        get_bpe_bliss_lexicon(bpe_size, add_blank=True, add_sentence_end_pron=False, add_oov_fallback=True),
        get_bpe_vocab_file(bpe_size, add_blank=True),
        binary,
        f"{prefix}/bpe",
        # Existing BPE lexicon calls the vocabulary's EOS column <s>. This is a
        # name alias at the same index, not an acoustic permutation.
        label_aliases={"</s>": "<s>"},
        decoder_smoke_report=smoke.out_report,
    )
    bpe_variants = prototype_variants("bpe")
    for variant in bpe_variants:
        variant.rasr_binary_path_override = binary
        variant.prepared_lexicon_file = bpe_lexicon
    results = ctc_bpe.run(ctc_model, variants=bpe_variants, corpora=["dev-other"])

    repo = get_huggingface_repo_dir(ctc_granite.GRANITE_CTC_REPO_ID, revision=ctc_granite.GRANITE_CTC_REVISION)
    inspection = InspectWordStartTokenizerJob(repo, datasets.get_default_score_corpus("dev-other").bliss_corpus_file)
    inspection.add_alias(f"{prefix}/granite/tokenizer")
    tk.register_output(f"{prefix}/granite/tokenizer-pieces", inspection.out_table)
    tk.register_output(f"{prefix}/granite/tokenizer-report", inspection.out_report)
    # The general closed-tree factory intentionally defaults to no unknown lemma.
    # Both prototype arms need the singleton LM-token anchor, even though it has
    # no acoustic pronunciation and introduces no additional acoustic label.
    original_granite_lexicon = HuggingFaceBpeBlissLexiconJob(
        get_bliss_lexicon(), repo, blank_index=0, add_unknown_lemma=True
    ).out_lexicon
    original_granite_lexicon = RemoveEmptyUnknownPronunciationsJob(original_granite_lexicon).out_lexicon
    fallback = BuildGraniteOovLexiconJob(original_granite_lexicon, inspection.out_table)
    tk.register_output(f"{prefix}/granite/lexicon-report", fallback.out_report)
    granite_lexicon = _validated_lexicon(
        fallback.out_lexicon,
        get_hf_ctc_vocab_file(repo, blank_index=0),
        binary,
        f"{prefix}/granite",
        decoder_smoke_report=smoke.out_report,
    )
    formatter = [
        Import(f"{GraniteOovTracebackFormatter.__module__}.{GraniteOovTracebackFormatter.__name__}"),
        Call(
            "GraniteOovTracebackFormatter",
            kwargs=[("huggingface_repo_dir", DelayedFormat('tk.Path("{}")', repo))],
            return_assign_variables="traceback_formatter",
        ),
    ]
    granite_variants = prototype_variants("granite")
    for variant in granite_variants:
        variant.rasr_binary_path_override = binary
        variant.prepared_lexicon_file = granite_lexicon
        variant.prototype_traceback_formatter_serializers = formatter
    # Measure the single-surface-word restriction itself, independently of the
    # fallback. These closed controls use the original lexicon on the SAME build.
    original_checked = _validated_lexicon(
        original_granite_lexicon,
        get_hf_ctc_vocab_file(repo, blank_index=0),
        binary,
        f"{prefix}/granite-original-lexicon",
        decoder_smoke_report=smoke.out_report,
    )
    for variant in list(granite_variants):
        p = variant.search_algorithm_params
        if (
            getattr(p, "unknown_word_fallback", None) == "disabled"
            and p.max_beam_sizes == [256]
            and (p.word_lm_params is None or p.word_lm_params.scale == 0.3)
        ):
            granite_variants.append(
                replace(
                    variant,
                    descriptor=variant.descriptor + "_original-lexicon",
                    prepared_lexicon_file=original_checked,
                )
            )
    results.extend(ctc_granite.run(huggingface_repo_dir=repo, variants=granite_variants, corpora=["dev-other"]))
    return results
