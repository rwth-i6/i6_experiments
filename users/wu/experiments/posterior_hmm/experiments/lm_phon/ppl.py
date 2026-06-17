"""
Perplexity evaluation for the phoneme LMs (neural Transformer + count n-gram) and the
official LibriSpeech word 4-gram, on dev-other / test-other transcripts.

Three measurement paths, all made mutually consistent (see the convention note below):

  * neural phoneme LM : RETURNN forward (`ppl_forward_v2`) over the phonemized eval text
                        -> ppl.json with total NLL (nats) + token counts, with/without </s>.
  * count phoneme LM  : RASR `lm-util compute-perplexity-from-text-file` over the SAME
                        phonemized eval text, scored against the count ARPA via an IDENTITY
                        phoneme lexicon.
  * official word LM  : RASR `lm-util` over the word eval text, scored against the official
                        4-gram ARPA with the standard LibriSpeech word lexicon (UNK-aware).

Reported metrics (all WITHOUT unknowns; the phoneme LMs are closed-vocab so that is a no-op):
  * phoneme PPL (+eos / -eos): per-phoneme-token perplexity (phoneme LMs only).
  * word PPL    (+eos / -eos): per-WORD perplexity. For the word LM this is its native
    (UNK-excluded) perplexity; for the phoneme LMs it is the WORD-EQUIVALENT perplexity
    `exp(total_neg_log_prob / N_words)`.

Convention (matches RASR lm-util exactly, verified against src/Tools/.../LmUtilityTool.cc):
  lm-util counts one `</s>` per sentence in `num_tokens`; `perplexity = exp(NLL/num_tokens)`
  (with-eos), `perplexity_without_eos = exp((NLL-eos)/(num_tokens-num_sent))`. The neural
  `LmDataset` appends `</s>` to `data`, so the neural with-eos number normalizes by
  `N_phon + N_sent` and without-eos by `N_phon` -- identical convention. NLLs are in nats.

Held-out-ness: test-other is fully unseen by all three LMs; dev-other is the neural/count
phoneme LMs' CV set (held out from the gradient but used for selection). The official 4-gram
is trained on Gutenberg text, disjoint from both. dev-other rows are labelled accordingly in
the report.
"""

import copy
import math
import os
import pickle
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

from sisyphus import Job, Task, tk

from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.g2p.apply import ApplyG2PModelJob
from i6_core.lib.lexicon import Lemma, Lexicon
from i6_core.lm.perplexity import ComputePerplexityJobV2
from i6_core.rasr import CommonRasrParameters, RasrConfig, crp_add_default_output
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJobV2
from i6_core.util import uopen, write_xml

from i6_experiments.common.datasets.librispeech import get_bliss_corpus_dict
from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict
from i6_experiments.common.datasets.librispeech.lexicon import get_bliss_lexicon

from ...config import get_forward_config
from ...data.phon import get_phmm_eow_lexicon, get_phmm_eow_lm_vocab_datastream
from ...data.phon_lm import (
    CollectOovWordsJob,
    LmDataset,
    TextToPhonemeJob,
    _train_g2p_model,
)
from ...default_tools import LM_UTIL_EXE, RETURNN_EXE, RETURNN_ROOT
from ...pipeline import NeuralLM
from ...ppl_report import add_eval_set_stats, add_ppl_result, set_report_path
from ...storage import get_lm_model
from .count_ngram import build_phon_count_ngram_lm


# ============================================================================
# Jobs
# ============================================================================


class PhonemeVocabToIdentityLexiconJob(Job):
    """
    Build an IDENTITY Bliss lexicon from a phoneme LM `vocab.pkl` so RASR `lm-util` can score a
    phoneme ARPA on a phoneme text file. Each real phoneme becomes a lemma with `orth == phon ==
    synt` (the phoneme maps 1:1 to its own LM token). `<s>`/`</s>` become the sentence-begin/end
    special lemmata, and a `[UNKNOWN]` special lemma is added because lm-util REQUIRES a special
    lemma named "unknown" (it aborts otherwise); the phoneme inventory is closed so it is never
    actually invoked (num_unks == 0).
    """

    def __init__(
        self,
        vocab: tk.Path,
        *,
        bos_token: str = "<s>",
        eos_token: str = "</s>",
        silence_token: str = "[SILENCE]",
        include_silence: bool = False,
    ):
        super().__init__()
        self.vocab = vocab
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.silence_token = silence_token
        self.include_silence = include_silence
        self.out_lexicon = self.output_path("identity_lexicon.xml.gz")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        with uopen(self.vocab.get_path(), "rb") as f:
            vocab: Dict[str, int] = pickle.load(f)

        specials = {self.bos_token, self.eos_token}
        real_tokens = [tok for tok, _ in sorted(vocab.items(), key=lambda kv: kv[1]) if tok not in specials]
        if not self.include_silence and self.silence_token in real_tokens:
            real_tokens = [t for t in real_tokens if t != self.silence_token]

        lex = Lexicon()
        for tok in real_tokens:
            lex.add_phoneme(tok, variation="none")

        # identity lemma per phoneme: surface form == pronunciation == LM token
        for tok in real_tokens:
            lex.add_lemma(Lemma(orth=[tok], phon=[tok], synt=[tok]))

        # sentence boundaries: lm-util scores </s> per sentence and starts from <s>.
        lex.add_lemma(Lemma(orth=[self.bos_token], synt=[self.bos_token], special="sentence-begin"))
        lex.add_lemma(Lemma(orth=[self.eos_token], synt=[self.eos_token], special="sentence-end"))
        # REQUIRED by lm-util even though it is never invoked for a closed phoneme vocab.
        lex.add_lemma(Lemma(orth=["[UNKNOWN]"], synt=["<UNK>"], special="unknown"))

        write_xml(self.out_lexicon.get_path(), lex.to_xml())


class CountTextStatsJob(Job):
    """Count non-empty lines (= sentences) and whitespace tokens of a (optionally gzipped) text file."""

    def __init__(self, text_file: tk.Path):
        super().__init__()
        self.text_file = text_file
        self.out_num_lines = self.output_var("num_lines")
        self.out_num_tokens = self.output_var("num_tokens")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        n_lines = 0
        n_tokens = 0
        with uopen(self.text_file.get_path(), "rt") as f:
            for line in f:
                toks = line.split()
                if not toks:
                    continue
                n_lines += 1
                n_tokens += len(toks)
        self.out_num_lines.set(n_lines)
        self.out_num_tokens.set(n_tokens)


class AssertPhonemizationCompleteJob(Job):
    """
    Fail loudly if eval-set phonemization dropped any sentence. The word-equivalent perplexity
    normalizes the phoneme LMs' total NLL by `N_words` counted on the FULL word text, so it is only
    exact if the phoneme LMs scored exactly the same sentences. `TextToPhonemeJob` drops sentences
    containing an unresolvable word, so a nonzero drop (= `num_unresolved > 0`, equivalently
    `phon_num_lines != word_num_lines`) would silently bias the word-equivalent numbers. With the
    per-eval-set G2P fallback this should never happen; if it ever does, this job surfaces the
    offending words instead of reporting a subtly-wrong perplexity.
    """

    def __init__(self, *, num_unresolved, word_num_lines, phon_num_lines, unresolved_words):
        super().__init__()
        self.num_unresolved = num_unresolved
        self.word_num_lines = word_num_lines
        self.phon_num_lines = phon_num_lines
        self.unresolved_words = unresolved_words
        self.out_ok = self.output_var("coverage_ok")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        n_unres = int(self.num_unresolved.get())
        n_word = int(self.word_num_lines.get())
        n_phon = int(self.phon_num_lines.get())
        if n_unres != 0 or n_word != n_phon:
            with uopen(self.unresolved_words.get_path(), "rt") as f:
                sample = [w.strip() for w in f.read().split() if w.strip()][:20]
            raise AssertionError(
                f"eval-set phonemization is incomplete: {n_unres} unresolvable word types, "
                f"word lines={n_word} vs phon lines={n_phon}. The word-equivalent perplexity would "
                f"be biased. Sample unresolved words: {sample}. Extend the G2P fallback or handle "
                f"the dropped sentences before trusting these numbers."
            )
        self.out_ok.set(True)


class NeuralPplMetricsJob(Job):
    """
    Turn the neural `ppl.json` (raw NLL + counts from `ppl_forward_v2`) into the reported metrics.
    word PPL = exp(total_neg_log_prob / N_words) (word-equivalent perplexity).
    """

    def __init__(self, ppl_json: tk.Path, n_words: tk.Variable):
        super().__init__()
        self.ppl_json = ppl_json
        self.n_words = n_words
        self.out_phon_ppl_eos = self.output_var("phon_ppl_eos")
        self.out_phon_ppl_noeos = self.output_var("phon_ppl_noeos")
        self.out_word_ppl_eos = self.output_var("word_ppl_eos")
        self.out_word_ppl_noeos = self.output_var("word_ppl_noeos")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import json

        with uopen(self.ppl_json.get_path(), "rt") as f:
            p = json.load(f)
        nw = float(self.n_words.get())
        self.out_phon_ppl_eos.set(float(p["perplexity_with_eos"]))
        self.out_phon_ppl_noeos.set(float(p["perplexity_without_eos"]))
        self.out_word_ppl_eos.set(math.exp(float(p["total_neg_log_prob_with_eos"]) / nw))
        self.out_word_ppl_noeos.set(math.exp(float(p["total_neg_log_prob_without_eos"]) / nw))


class RasrPplMetricsJob(Job):
    """
    Turn RASR `lm-util` perplexity outputs into the reported metrics.

    For a PHONEME ARPA (is_word_lm=False): per-phoneme PPL == RASR perplexity[/without_eos];
    word-equivalent PPL = exp(NLL / N_words) with NLL backed out as num_tokens*ln(ppl) (with-eos)
    and (num_tokens-N_sent)*ln(ppl_without_eos) (without-eos).

    For the WORD ARPA (is_word_lm=True): uses the WITHOUT-UNKNOWNS variants; word PPL =
    exp(NLL_nounk / N_words) with NLL_nounk = (num_tokens-num_unks)*ln(ppl_without_unknowns)
    (with-eos) and (num_tokens-num_unks-N_sent)*ln(ppl_without_eos_without_unknowns) (without-eos).
    """

    def __init__(
        self,
        *,
        num_tokens: tk.Variable,
        n_words: tk.Variable,
        n_sent: tk.Variable,
        is_word_lm: bool,
        perplexity: Optional[tk.Variable] = None,
        perplexity_without_eos: Optional[tk.Variable] = None,
        perplexity_without_unknowns: Optional[tk.Variable] = None,
        perplexity_without_eos_without_unknowns: Optional[tk.Variable] = None,
        num_unks: Optional[tk.Variable] = None,
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.n_words = n_words
        self.n_sent = n_sent
        self.is_word_lm = is_word_lm
        self.perplexity = perplexity
        self.perplexity_without_eos = perplexity_without_eos
        self.perplexity_without_unknowns = perplexity_without_unknowns
        self.perplexity_without_eos_without_unknowns = perplexity_without_eos_without_unknowns
        self.num_unks = num_unks

        self.out_word_ppl_eos = self.output_var("word_ppl_eos")
        self.out_word_ppl_noeos = self.output_var("word_ppl_noeos")
        if not is_word_lm:
            self.out_phon_ppl_eos = self.output_var("phon_ppl_eos")
            self.out_phon_ppl_noeos = self.output_var("phon_ppl_noeos")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        nt = float(self.num_tokens.get())
        nw = float(self.n_words.get())
        ns = float(self.n_sent.get())

        if not self.is_word_lm:
            p_eos = float(self.perplexity.get())
            p_noeos = float(self.perplexity_without_eos.get())
            nll_eos = nt * math.log(p_eos)
            nll_noeos = (nt - ns) * math.log(p_noeos)
            self.out_phon_ppl_eos.set(p_eos)
            self.out_phon_ppl_noeos.set(p_noeos)
            self.out_word_ppl_eos.set(math.exp(nll_eos / nw))
            self.out_word_ppl_noeos.set(math.exp(nll_noeos / nw))
        else:
            nu = float(self.num_unks.get())
            p_wu = float(self.perplexity_without_unknowns.get())
            p_weu = float(self.perplexity_without_eos_without_unknowns.get())
            nll_eos = (nt - nu) * math.log(p_wu)
            nll_noeos = (nt - nu - ns) * math.log(p_weu)
            self.out_word_ppl_eos.set(math.exp(nll_eos / nw))
            self.out_word_ppl_noeos.set(math.exp(nll_noeos / nw))


# ============================================================================
# Helpers
# ============================================================================


def _build_eval_phoneme_text(
    prefix: str,
    eval_set: str,
    word_text: tk.Path,
    bliss_lexicon: tk.Path,
    g2p_model: tk.Path,
) -> TextToPhonemeJob:
    """
    Phonemize the eval word text with the SAME pHMM lexicon + Sequitur G2P as LM training, so the
    tokens are in the LM's phoneme inventory. A per-eval-set G2P over the eval OOVs guarantees no
    sentence is dropped (the caller asserts this via AssertPhonemizationCompleteJob).
    """
    oov = CollectOovWordsJob(text_files=[word_text], bliss_lexicon=bliss_lexicon)
    oov.add_alias(os.path.join(prefix, "ppl", eval_set, "collect_oov"))
    g2p_apply = ApplyG2PModelJob(
        g2p_model=g2p_model,
        word_list_file=oov.out_word_list,
        filter_empty_words=True,
        concurrent=4,
    )
    g2p_apply.add_alias(os.path.join(prefix, "ppl", eval_set, "apply_g2p"))
    phon = TextToPhonemeJob(
        text_file=word_text,
        bliss_lexicon=bliss_lexicon,
        g2p_lexicon=g2p_apply.out_g2p_lexicon,
        gzip_output=True,
    )
    phon.add_alias(os.path.join(prefix, "ppl", eval_set, "phonemize"))
    return phon


def _arpa_ppl_crp(arpa_lm: tk.Path, lexicon_file: tk.Path) -> CommonRasrParameters:
    """Build a CommonRasrParameters for lm-util perplexity over an ARPA LM (no LM image needed)."""
    lm_config = RasrConfig()
    lm_config.type = "ARPA"
    lm_config.file = arpa_lm
    lm_config.scale = 1.0

    crp = CommonRasrParameters()
    crp_add_default_output(crp)
    crp.lm_util_exe = LM_UTIL_EXE
    crp.language_model_config = lm_config
    crp.lexicon_config = RasrConfig()
    crp.lexicon_config.file = lexicon_file
    crp.lexicon_config.normalize_pronunciation = False
    return crp


def _run_neural_ppl(
    *,
    prefix_name: str,
    lm_model: NeuralLM,
    label_datastream,
    phon_text: tk.Path,
) -> ReturnnForwardJobV2:
    """RETURNN forward of the neural LM over an LmDataset built from `phon_text` -> ppl.json."""
    eval_dataset = LmDataset(
        corpus_file=phon_text,
        vocab_file=lm_model.phon_vocab,
        partition_epoch=1,
        segment_file=None,
        seq_ordering="sorted",
    )
    label_opts = label_datastream.as_returnn_extern_data_opts(available_for_inference=True)
    forward_config = {
        "behavior_version": 21,
        "extern_data": {
            "data": {**label_opts, "available_for_inference": True},
            "delayed": {**label_opts, "available_for_inference": True},
        },
        "batch_size": 4000,
        "max_seqs": 64,
        "torch_amp": {"dtype": "bfloat16"},
        "torch_dataloader_opts": {"num_workers": 1},
    }
    returnn_config: ReturnnConfig = get_forward_config(
        network_module=lm_model.network_module,
        config=forward_config,
        net_args=lm_model.net_args,
        decoder="lm.trafo.ppl_forward_v2",
        decoder_args={},
        # LmDataset wraps corpus_file in cf(...); without the cache-manager prolog the dataloader
        # worker raises NameError: name 'cf' is not defined (same as LM training, which sets this).
        add_cache_manager=True,
        debug=False,
    )
    returnn_config = copy.deepcopy(returnn_config)
    # real RETURNN reads the forward dataset from the `forward_data` key (not `forward`).
    returnn_config.config["forward_data"] = eval_dataset.as_returnn_opts()

    ppl_job = ReturnnForwardJobV2(
        model_checkpoint=lm_model.checkpoint,
        returnn_config=returnn_config,
        log_verbosity=5,
        mem_rqmt=16,
        time_rqmt=4,
        device="gpu",
        cpu_rqmt=2,
        returnn_python_exe=RETURNN_EXE,
        returnn_root=RETURNN_ROOT,
        output_files=["ppl.json"],
    )
    # Request a 24 GB GPU so this lands on the gpu_24gb partition (where training reliably runs)
    # instead of the flaky default gpu_11gb pool, where nvidia-smi + a current driver are present
    # but torch's cuInit() fails ("No GPU device found, but config requested 'gpu'"). rqmt is not
    # hashed, so this does not re-key the job.
    ppl_job.rqmt["gpu_mem"] = 24
    ppl_job.add_alias(prefix_name + "/ppl_job")
    tk.register_output(prefix_name + "/ppl.json", ppl_job.out_files["ppl.json"])
    return ppl_job


# ============================================================================
# Top-level wiring
# ============================================================================


def evaluate_lm_perplexities(
    prefix: str = "example_setups/librispeech/posterior_hmm/lm_phon",
    *,
    librispeech_key: str = "train-other-960",
    eval_keys: Tuple[str, ...] = ("dev-other", "test-other"),
    neural_lm_name: str = "phon_trafo12x512_3ep",
    count_order: int = 8,
    count_pruning: Optional[List[int]] = None,
):
    """
    Wire up phoneme + word perplexity for the neural Transformer phoneme LM, the count n-gram
    phoneme LM and the official LibriSpeech word 4-gram, on `eval_keys`, into one live report
    at `<prefix>/perplexity.report`.

    :param count_order: order of the count phoneme LM to evaluate.
    :param count_pruning: per-order count pruning of the count LM. MUST match the value used in the
        recognition config (e.g. config_01's ``lexfree_count_eow_phon_phmm_ls960(pruning=...)``) so
        the SAME ARPA is reused by hash rather than building a second (unpruned) one.
    """
    set_report_path(prefix + "/perplexity.report")

    # Shared phonemization machinery (identical jobs as LM training -> reused by hash).
    bliss_lexicon = get_phmm_eow_lexicon(g2p_librispeech_key=librispeech_key)
    g2p_model = _train_g2p_model(prefix=prefix, bliss_lexicon=bliss_lexicon)
    lm_vocab_ds = get_phmm_eow_lm_vocab_datastream(prefix=prefix, g2p_librispeech_key=librispeech_key)

    # LMs. Train + register the neural Transformer LM here so the perplexity eval is self-contained
    # (it no longer rides on the recognition path, which previously triggered this registration).
    from .trafo import phon_trafo_12x512_baseline

    phon_trafo_12x512_baseline()
    neural_lm: NeuralLM = get_lm_model(neural_lm_name)
    assert neural_lm.phon_vocab is not None, f"NeuralLM {neural_lm_name!r} has no phon_vocab"
    count_lm = build_phon_count_ngram_lm(
        prefix=prefix, librispeech_key=librispeech_key, order=count_order, pruning=count_pruning
    )
    count_arpa = count_lm["arpa"]
    word_arpa = get_arpa_lm_dict()["4gram"]

    # Lexicons for lm-util.
    identity_lexicon = PhonemeVocabToIdentityLexiconJob(vocab=lm_vocab_ds.vocab).out_lexicon
    word_lexicon = get_bliss_lexicon(
        use_stress_marker=False,
        add_unknown_phoneme_and_mapping=True,  # required so OOV words map to <UNK>
        add_silence=True,
    )

    bliss_corpus_dict = get_bliss_corpus_dict()

    def _emit(eval_set, eval_label, lm_slug, lm_label, order, metrics):
        # Register every metric Variable as an output so the manager actually RUNS the upstream
        # ComputePerplexityJob/metrics jobs. A report's `required=` references the variables but does
        # NOT pull their jobs into execution (unlike results.py, where each WER is register_output'd
        # in pipeline.py) -- without this the report cells stay '--' forever.
        for key, var in metrics.items():
            tk.register_output(os.path.join(prefix, "ppl", eval_set, "metrics", lm_slug, key), var)
        add_ppl_result(eval_label, lm_label, order=order, metrics=metrics)

    for eval_set in eval_keys:
        word_text = CorpusToTxtJob(bliss_corpus=bliss_corpus_dict[eval_set], gzip=True).out_txt
        phon_job = _build_eval_phoneme_text(
            prefix=prefix,
            eval_set=eval_set,
            word_text=word_text,
            bliss_lexicon=bliss_lexicon,
            g2p_model=g2p_model,
        )
        phon_text = phon_job.out_text

        word_stats = CountTextStatsJob(word_text)
        word_stats.add_alias(os.path.join(prefix, "ppl", eval_set, "word_stats"))
        phon_stats = CountTextStatsJob(phon_text)
        phon_stats.add_alias(os.path.join(prefix, "ppl", eval_set, "phon_stats"))
        n_words = word_stats.out_num_tokens
        n_sent = word_stats.out_num_lines

        # Fail loudly if any sentence was dropped during phonemization: the word-equivalent PPL
        # normalizes by N_words over the FULL word text, so it is only exact if the phoneme LMs
        # scored every sentence (no unresolvable words). Forces the check into the graph.
        coverage = AssertPhonemizationCompleteJob(
            num_unresolved=phon_job.out_num_unresolved,
            word_num_lines=word_stats.out_num_lines,
            phon_num_lines=phon_stats.out_num_lines,
            unresolved_words=phon_job.out_unresolved,
        )
        tk.register_output(
            os.path.join(prefix, "ppl", eval_set, "phonemization_coverage_ok"), coverage.out_ok
        )

        add_eval_set_stats(
            eval_set,
            n_words=n_words,
            n_phon=phon_stats.out_num_tokens,
            n_sent=n_sent,
        )

        eval_label = eval_set + (" (CV)" if eval_set in ("dev-clean", "dev-other") else "")

        # --- neural phoneme LM ---
        neural_prefix = os.path.join(prefix, "ppl", eval_set, "neural_" + neural_lm_name)
        neural_job = _run_neural_ppl(
            prefix_name=neural_prefix,
            lm_model=neural_lm,
            label_datastream=lm_vocab_ds,
            phon_text=phon_text,
        )
        neural_metrics = NeuralPplMetricsJob(ppl_json=neural_job.out_files["ppl.json"], n_words=n_words)
        _emit(
            eval_set,
            eval_label,
            "neural_" + neural_lm_name,
            "neural trafo 12x512",
            0,
            {
                "phon_ppl_eos": neural_metrics.out_phon_ppl_eos,
                "phon_ppl_noeos": neural_metrics.out_phon_ppl_noeos,
                "word_ppl_eos": neural_metrics.out_word_ppl_eos,
                "word_ppl_noeos": neural_metrics.out_word_ppl_noeos,
            },
        )

        # --- count phoneme LM (RASR lm-util over the phoneme text + identity lexicon) ---
        count_crp = _arpa_ppl_crp(count_arpa, identity_lexicon)
        count_ppl = ComputePerplexityJobV2(crp=count_crp, text_file=phon_text)
        count_ppl.add_alias(os.path.join(prefix, "ppl", eval_set, f"count_{count_order}gram_ppl"))
        # The high-order phoneme ARPA is large (~73M n-grams for the pruned 8-gram); RASR's ARPA
        # reader loads it fully (~11+ GB measured, loads in ~2 min), so the i6_core default mem=2 OOMs.
        count_ppl.rqmt = {"time": 2, "cpu": 1, "mem": 32}
        count_metrics = RasrPplMetricsJob(
            num_tokens=count_ppl.num_tokens,
            n_words=n_words,
            n_sent=phon_stats.out_num_lines,
            is_word_lm=False,
            perplexity=count_ppl.perplexity,
            perplexity_without_eos=count_ppl.perplexity_without_eos,
        )
        _emit(
            eval_set,
            eval_label,
            f"count_{count_order}gram",
            f"count {count_order}-gram",
            1,
            {
                "phon_ppl_eos": count_metrics.out_phon_ppl_eos,
                "phon_ppl_noeos": count_metrics.out_phon_ppl_noeos,
                "word_ppl_eos": count_metrics.out_word_ppl_eos,
                "word_ppl_noeos": count_metrics.out_word_ppl_noeos,
            },
        )

        # --- official word 4-gram (RASR lm-util over the word text + LibriSpeech word lexicon) ---
        word_crp = _arpa_ppl_crp(word_arpa, word_lexicon)
        word_ppl = ComputePerplexityJobV2(crp=word_crp, text_file=word_text)
        word_ppl.add_alias(os.path.join(prefix, "ppl", eval_set, "word_4gram_ppl"))
        # The official LibriSpeech 4-gram ARPA is large (~145M n-grams); RASR's ARPA reader loads it
        # fully into RAM, so the i6_core default (mem=2) OOMs. rqmt is not hashed, so this is a free
        # bump. (The count phoneme 8-gram is smaller, ~11-14 GB -> 24 GB above.)
        word_ppl.rqmt = {"time": 4, "cpu": 1, "mem": 48}
        word_metrics = RasrPplMetricsJob(
            num_tokens=word_ppl.num_tokens,
            n_words=n_words,
            n_sent=n_sent,
            is_word_lm=True,
            perplexity_without_unknowns=word_ppl.perplexity_without_unknowns,
            perplexity_without_eos_without_unknowns=word_ppl.perplexity_without_eos_without_unknowns,
            num_unks=word_ppl.num_unks,
        )
        _emit(
            eval_set,
            eval_label,
            "word_4gram",
            "official word 4-gram",
            2,
            {
                "word_ppl_eos": word_metrics.out_word_ppl_eos,
                "word_ppl_noeos": word_metrics.out_word_ppl_noeos,
            },
        )
