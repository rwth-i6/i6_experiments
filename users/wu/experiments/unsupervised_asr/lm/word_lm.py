"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/lexlat_jobs.py (``LexiconTrieBuildJob`` and
its constants), with the resource wiring of
src/speech_llm/prefix_lm/sis_recipe/exp2025_11_06_speech_llms/librispeech/configs/
config_sae_4a_lexlat_probes_v1.py (``resource_job``) and config_sae_4a_lexlat_k2_official_v1.py
(``official_lms``, ``OFFICIAL_LEXICON``).

The word side of phase 4a:

* the official openslr 11 resources -- the word 4-gram (``i6_experiments.common``
  ``get_arpa_lm_dict()["4gram"]``), the 3-gram pruned 1e-7 (a ``DownloadJob`` with its sha256), the
  LM-norm text (``get_librispeech_normalized_lm_data()``) and the raw ``librispeech-lexicon.txt``
  (the ``DownloadJob`` ``i6_experiments.common``'s lexicon module itself runs, same arguments, so
  the same job);
* :class:`LexiconTrieBuildJob` -- the in-house trie + word-trigram CSR automaton + escape prices
  + the null's derangement, as one ``lexlat_resources.npz``; and :func:`get_lexicon_trie`, which
  wires it from the public inputs.

Port change: the source job replayed the window and ran ``lmplz`` / ``build_binary`` itself.  In the
port the replay is :class:`.word_window.WordWindowReplayJob`, and the word trigram is i6_core's
``KenLMplzJob`` + ``CreateBinaryLMJob`` (KenLM from ``default_tools.get_kenlm_binary_path()``), set up
by :func:`get_lexicon_trie` to issue the source's own command lines: ``lmplz -o 3
--interpolate_unigrams True -S 24G -T <tmp> --discount_fallback 0.5 1.0 1.5`` on the window's word
lines, then ``build_binary <arpa> <binary>`` with no options (the default probing format).  Only the
``-T`` scratch directory differs, and the ARPA reaches ``build_binary`` gzipped; on a 200,000-line
LibriSpeech LM text both flows gave byte-identical ARPA text and binaries.  ``KenLMplzJob`` derives
``-S`` from its memory request, so that job requests 24 GB, the ``-S`` the source passed.

Cut from the source module: ``LexlatEquivalenceProbeJob`` (E-1) and ``LexlatCensusJob`` (E0), the
two finished pre-funding probes, which no in-scope run uses.
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
from typing import Dict, List, Optional, Sequence

from sisyphus import Job, Task, tk

__all__ = [
    "RESOURCE_NPZ",
    "WORD_LM_BIN",
    "LEXICON_JSON",
    "SHUFFLED_JSON",
    "BANKED_WORDS",
    "BANKED_BIGRAM_TYPES",
    "BANKED_DISTINCT_PRONUNCIATIONS",
    "BANKED_MAX_PRONUNCIATION",
    "DERANGEMENT_SEED",
    "PHONES_PER_WORD_BAND",
    "URL_3G_1E7",
    "SHA256_3G_1E7",
    "official_4gram_arpa",
    "official_3gram_pruned_1e7_arpa",
    "official_lexicon",
    "WORD_LM_ORDER",
    "LMPLZ_MEM_GB",
    "LexiconTrieBuildJob",
    "get_lexicon_trie",
]


#: the file names of the resource directory every consumer of this phase reads
RESOURCE_NPZ = "lexlat_resources.npz"
WORD_LM_BIN = "word_lm.bin"
LEXICON_JSON = "lexicon.json.gz"
SHUFFLED_JSON = "shuffled_lexicon.json.gz"

#: the banked identity of the Step 0 resource (survey ``survey_lexicon_scorer_2026-09-20.md`` s1,
#: s4; `SAE_4A_lexlat.md` Design 8).  These are ASSERTED, not measured here.
BANKED_WORDS = 151_731
BANKED_BIGRAM_TYPES = 3_302_936
#: the survey's other two resource numbers, asserted beside them
BANKED_DISTINCT_PRONUNCIATIONS = 132_049
BANKED_MAX_PRONUNCIATION = 33
#: the seed of the shuffled-pronunciation null (Design 6: "a fixed-seed derangement")
DERANGEMENT_SEED = 0
#: Design 4's monitor band, as a multiple of the token-weighted mean pronunciation length
PHONES_PER_WORD_BAND = (0.7, 1.4)
#: the in-house word LM's order (the source job's ``word_lm_order`` default)
WORD_LM_ORDER = 3
#: ``lmplz -S`` of the source call (half the source job's 48 GB); ``KenLMplzJob`` takes it as ``mem``
LMPLZ_MEM_GB = 24


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")

# ---------------------------------------------------------------------------------------------------
# the official openslr 11 resources
# ---------------------------------------------------------------------------------------------------

#: the official 3-gram pruned 1e-7, which ``i6_experiments.common`` does not list
URL_3G_1E7 = "https://www.openslr.org/resources/11/3-gram.pruned.1e-7.arpa.gz"
#: sha256 of the file the banked ``DownloadJob.n2emmqhiewWZ`` (checksum None) fetched from that URL
SHA256_3G_1E7 = "71dd880667cb6ea3942bb1982988c91fb1208839a21911eeec7e31c857c4bdfb"


def official_4gram_arpa() -> tk.Path:
    """The official word 4-gram ``4-gram.arpa.gz`` (``i6_experiments.common``, sha256-checked)."""
    from i6_experiments.common.datasets.librispeech.language_model import get_arpa_lm_dict

    return get_arpa_lm_dict()["4gram"]


def official_3gram_pruned_1e7_arpa() -> tk.Path:
    """The official ``3-gram.pruned.1e-7.arpa.gz``."""
    from i6_core.tools.download import DownloadJob

    job = DownloadJob(url=URL_3G_1E7, target_filename=URL_3G_1E7.split("/")[-1],
                      checksum=SHA256_3G_1E7)
    job.add_alias("sae/4a/lexlat_k2_official/download/3gram_pruned_1e-7")
    return job.out_file


def official_lexicon() -> tk.Path:
    """The raw openslr ``librispeech-lexicon.txt``.

    ``i6_experiments.common.datasets.librispeech.lexicon._get_raw_bliss_lexicon`` downloads it with
    exactly these arguments (and converts it to bliss); the text file itself is what
    ``lexlat_k2_official.read_official_lexicon`` reads, so the same ``DownloadJob`` is taken here.
    """
    from i6_core.tools.download import DownloadJob

    return DownloadJob(
        url="https://www.openslr.org/resources/11/librispeech-lexicon.txt",
        target_filename="librispeech-lexicon.txt",
        checksum="d722bc29908cd338ae738edd70f61826a6fca29aaa704a9493f0006773f79d71",
    ).out_file


# ---------------------------------------------------------------------------------------------------
# the resource
# ---------------------------------------------------------------------------------------------------

class LexiconTrieBuildJob(Job):
    """Build the trie, the word-trigram CSR automaton, the escape prices and the null's derangement.

    THE WORD SET IS NOT CHOSEN HERE.  ``prior_gap.load_phonemization_lexicon`` (the bliss lexicon
    plus the Sequitur g2p entries, exactly ``PhonemizeWithSilJob``'s own map) and
    ``prior_gap.restrict_to_word_lm`` (only the words the word LM has in its vocabulary -- without
    it an unseen word is a one-``<unk>`` cover for any phone span, review F2) are called verbatim,
    so the trie word set IS the banked Step 0 row's.

    PRE-REGISTERED CHECKS (Design 8), asserted:

      * ``|V| = 151,731`` -- the trie word count after ``restrict_to_word_lm``, and the word-type
        count of the replayed window text, which the survey measured to be the same set;
      * ``3,302,936`` distinct word bigram types on that text.  The survey counted them on the TEXT
        (``PriorGapAnalysisJob.pg14aEYJyiva/work/lms/window.words.txt``), not off the ARPA header,
        and the two differ by KenLM's ``<s>`` / ``</s>`` contexts, so three counting conventions
        are computed (in-line pairs; with ``<s>``; with ``<s>`` and ``</s>``) and the check is that
        the banked number is EXACTLY one of them.  The matching convention is recorded.

    PRINTED RESOURCE CONSTANTS (Design 4 and 8; neither is a result and neither is banked by the
    survey): the trie NODE COUNT, and the TOKEN-WEIGHTED MEAN PRONUNCIATION LENGTH -- the mean over
    the window text's word TOKENS, not over the vocabulary -- from which the anti-deletion monitor
    band ``lexlat_expected_phones_per_word in [0.7, 1.4] x mean`` is printed.  The type-weighted
    mean is printed beside it so the two are never confused.

    THE NULL.  ``lexlat.derange_pronunciations`` (Sattolo, seed 0) is applied to the SAME word set
    and the resulting trie is asserted to be bit-identical in ``child``, ``word_start`` and
    ``is_word_end`` -- the null moves ``word_id`` and nothing else, which is what makes it the
    prior-weight control of Design 6.  ``kept_by_homophony`` (words whose shuffled pronunciation is
    a homophone of their own) is the honest residual and is reported.

    :param bliss_lexicon: ``PhonemizeWithSilJob``'s own bliss lexicon.
    :param g2p_lexicon: its Sequitur g2p lexicon (``None`` = bliss only, the E-1 (a) variant).
    :param window_words: ``WordWindowReplayJob.out_words``, the window's counted word lines.
    :param window_stats: ``WordWindowReplayJob.out_stats``, the replay's counts (recorded).
    :param word_lm_arpa: the word trigram's gzipped ARPA (``KenLMplzJob.out_lm`` on ``window_words``).
    :param word_lm_binary: its KenLM binary (``CreateBinaryLMJob.out_lm``).
    :param prior_npz: the arm's own prior; its ORDER-1 row is the escape's per-phone price.
    :param word_lm_order: the order the ARPA was trained with (recorded).

    The source job's ``word_corpus``, ``window_phn``, ``n_window_lines``, ``held_stride`` and
    ``sample_seed`` are :class:`.word_window.WordWindowReplayJob`'s now, and ``kenlm_binaries`` is the
    two i6_core jobs' (module docstring).
    """

    __sis_version__ = 1

    def __init__(
        self,
        *,
        bliss_lexicon: tk.Path,
        g2p_lexicon: Optional[tk.Path],
        window_words: tk.Path,
        window_stats: tk.Path,
        word_lm_arpa: tk.Path,
        word_lm_binary: tk.Path,
        prior_npz: tk.Path,
        word_lm_order: int = WORD_LM_ORDER,
        expected_words: int = BANKED_WORDS,
        expected_bigram_types: int = BANKED_BIGRAM_TYPES,
        derangement_seed: int = DERANGEMENT_SEED,
    ):
        super().__init__()
        self.bliss_lexicon = bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.window_words = window_words
        self.window_stats = window_stats
        self.word_lm_arpa = word_lm_arpa
        self.word_lm_binary = word_lm_binary
        self.prior_npz = prior_npz
        self.word_lm_order = int(word_lm_order)
        self.expected_words = int(expected_words)
        self.expected_bigram_types = int(expected_bigram_types)
        self.derangement_seed = int(derangement_seed)
        self.out_resources = self.output_path(RESOURCE_NPZ)
        self.out_word_lm = self.output_path(WORD_LM_BIN)
        self.out_lexicon = self.output_path(LEXICON_JSON)
        self.out_shuffled = self.output_path(SHUFFLED_JSON)
        self.out_summary = self.output_path("summary.txt")
        self.out_stats = self.output_path("build.json")
        #: the source job's request (it also ran the replay and lmplz); the ARPA of a 151,731-word
        #: trigram is parsed into arrays here, hence the memory
        self.rqmt = {"cpu": 4, "mem": 48, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    @staticmethod
    def _bigram_types(text_path: str) -> Dict[str, object]:
        """Distinct word bigram types under THREE conventions, plus the token and type counts."""
        import numpy as np

        ids: Dict[str, int] = {}
        counts: Dict[int, int] = {}
        plain: List[np.ndarray] = []
        bos: List[np.ndarray] = []
        eos: List[np.ndarray] = []
        n_tokens = n_lines = 0
        with open(text_path) as fh:
            for line in fh:
                words = line.split()
                if not words:
                    continue
                n_lines += 1
                n_tokens += len(words)
                row = np.empty(len(words), dtype=np.int64)
                for i, w in enumerate(words):
                    k = ids.setdefault(w, len(ids))
                    row[i] = k
                    counts[k] = counts.get(k, 0) + 1
                plain.append(np.stack([row[:-1], row[1:]], axis=1) if len(row) > 1
                             else np.zeros((0, 2), dtype=np.int64))
                bos.append(np.stack([np.full(1, -1, dtype=np.int64), row[:1]], axis=1))
                eos.append(np.stack([row[-1:], np.full(1, -2, dtype=np.int64)], axis=1))
        v = len(ids) + 2

        def _uniq(parts):
            pairs = np.concatenate(parts, axis=0)
            return int(np.unique((pairs[:, 0] + 2) * v + (pairs[:, 1] + 2)).size)

        return {
            "in_line": _uniq(plain),
            "with_bos": _uniq(plain + bos),
            "with_bos_eos": _uniq(plain + bos + eos),
            "n_tokens": n_tokens,
            "n_lines": n_lines,
            "n_types": len(ids),
            "token_counts": {w: counts[k] for w, k in ids.items()},
        }

    def run(self):
        import time

        import numpy as np

        from ..model import lexlat
        from ..phones import PHONE2ID, PHONES, SIL
        from .phone_prior import PhoneNgramPrior
        from .prior_gap import (
            KenLMWordLM, load_phonemization_lexicon, restrict_to_word_lm, witten_bell_utt_log_prob,
        )

        started = time.monotonic()
        out_dir = os.path.abspath("lms")
        os.makedirs(out_dir, exist_ok=True)

        w2p_all = load_phonemization_lexicon(
            self.bliss_lexicon.get_path(),
            None if self.g2p_lexicon is None else self.g2p_lexicon.get_path(),
        )
        with open(self.window_stats.get_path()) as fh:
            window = json.load(fh)
        print(f"window: {window['counted_lines']} counted lines", flush=True)
        # the plain ARPA text lmplz wrote (KenLMplzJob gzips it; the parser below reads plain text)
        arpa = os.path.join(out_dir, f"words_o{self.word_lm_order}.arpa")
        with gzip.open(self.word_lm_arpa.get_path(), "rb") as src, open(arpa, "wb") as dst:
            shutil.copyfileobj(src, dst)
        binary = self.word_lm_binary.get_path()
        word_lm = KenLMWordLM(binary)
        in_vocab = restrict_to_word_lm(w2p_all, word_lm)
        text = self._bigram_types(self.window_words.get_path())

        # --- the pre-registered identity checks --------------------------------------------------
        assert len(in_vocab) == self.expected_words, (
            f"the trie holds {len(in_vocab)} words, the banked Step 0 row holds "
            f"{self.expected_words}: this is NOT the banked resource")
        assert text["n_types"] == self.expected_words, (
            f"the window text has {text['n_types']} word types, not {self.expected_words}")
        matched = [k for k in ("in_line", "with_bos", "with_bos_eos")
                   if text[k] == self.expected_bigram_types]
        assert matched, (
            f"no bigram-counting convention reproduces the banked {self.expected_bigram_types} "
            f"distinct types: in_line={text['in_line']}, with_bos={text['with_bos']}, "
            f"with_bos_eos={text['with_bos_eos']}")

        # --- the trie, the automaton, the escape prices -------------------------------------------
        lm = lexlat.parse_arpa_word_lm(arpa)
        word2id = {w: i for i, w in enumerate(lm["words"])}
        trie = lexlat.build_trie(in_vocab, PHONE2ID, word2id, n_phones=len(PHONES), sil=SIL)
        shuffled, null_stats = lexlat.derange_pronunciations(in_vocab, seed=self.derangement_seed)
        trie_null = lexlat.build_trie(shuffled, PHONE2ID, word2id, n_phones=len(PHONES), sil=SIL)
        for key in ("child", "word_start", "is_word_end"):
            assert np.array_equal(trie[key], trie_null[key]), (
                f"the derangement moved the trie's {key}: it is not the prior-weight control")
        assert not np.array_equal(trie["word_id"], trie_null["word_id"]), "the null moved nothing"

        prior = PhoneNgramPrior.load(self.prior_npz.get_path())
        escape = {p: witten_bell_utt_log_prob(prior, [p], 1) for p in PHONES if p != SIL}
        escape_vec = np.zeros(len(PHONES), dtype=np.float64)
        for phone, k in PHONE2ID.items():
            escape_vec[k] = float(escape.get(phone, 0.0))

        lexlat.save_resources(
            self.out_resources.get_path(), trie=trie, lm=lm, escape_phone_log_prob=escape_vec,
            n_phones=len(PHONES), sil_id=int(PHONE2ID[SIL]),
            shuffled_word_id=trie_null["word_id"])
        shutil.copyfile(binary, self.out_word_lm.get_path())
        for path, payload in ((self.out_lexicon, in_vocab), (self.out_shuffled, shuffled)):
            with gzip.open(path.get_path(), "wt") as fh:
                json.dump({w: list(p) for w, p in sorted(payload.items())}, fh)

        # --- the resource constants ----------------------------------------------------------------
        lens = {w: len(p) for w, p in in_vocab.items()}
        tok = text["token_counts"]
        n_tok = sum(tok[w] for w in lens)
        mean_token = sum(tok[w] * lens[w] for w in lens) / n_tok
        mean_type = _mean(list(lens.values()))
        distinct = len({tuple(p) for p in in_vocab.values()})
        record = {
            "words": len(in_vocab),
            "phonemization_words": len(w2p_all),
            "distinct_pronunciations": distinct,
            "max_pronunciation": max(lens.values()),
            "trie_nodes": int(trie["n_nodes"]),
            "trie_entries": int(trie["n_entries"]),
            "mean_pronunciation_token_weighted": float(mean_token),
            "mean_pronunciation_type_weighted": float(mean_type),
            "phones_per_word_band": [float(PHONES_PER_WORD_BAND[0] * mean_token),
                                     float(PHONES_PER_WORD_BAND[1] * mean_token)],
            "lm": {"order": self.word_lm_order, "n_states": int(lm["n_states"]),
                   "n_words": int(lm["n_words"]), "n_1gram": int(lm["n_1gram"]),
                   "n_2gram": int(lm["n_2gram"]), "n_3gram": int(lm["n_3gram"]),
                   "binary_bytes": os.path.getsize(binary)},
            "text": {k: text[k] for k in ("in_line", "with_bos", "with_bos_eos", "n_tokens",
                                          "n_lines", "n_types")},
            "bigram_convention_matched": matched,
            "expected": {"words": self.expected_words,
                         "bigram_types": self.expected_bigram_types},
            "null": null_stats,
            "escape_phone_log_probs": {p: float(v) for p, v in escape.items()},
            "window": {k: window[k] for k in sorted(window) if isinstance(window[k], (int, float))},
            "runtime_seconds": time.monotonic() - started,
        }
        with open(self.out_stats.get_path(), "w") as fh:
            json.dump(record, fh, indent=2, sort_keys=True)
        lines = [
            "SAE 4a lexlat -- LexiconTrieBuildJob (the phase's frozen resource)",
            "",
            f"words in the trie                  {record['words']} "
            f"(banked {self.expected_words}; {record['phonemization_words']} phonemization words)",
            f"distinct pronunciations            {record['distinct_pronunciations']} "
            f"(survey {BANKED_DISTINCT_PRONUNCIATIONS})",
            f"max pronunciation length           {record['max_pronunciation']} "
            f"(survey {BANKED_MAX_PRONUNCIATION})",
            f"TRIE NODES                         {record['trie_nodes']} "
            f"({record['trie_entries']} word ends)",
            f"MEAN PRONUNCIATION, token-weighted {mean_token:.4f} phones "
            f"(type-weighted {mean_type:.4f})",
            f"  -> Design 4 monitor band: lexlat_expected_phones_per_word in "
            f"[{record['phones_per_word_band'][0]:.4f}, {record['phones_per_word_band'][1]:.4f}]",
            "",
            f"word LM order {self.word_lm_order}: {record['lm']['n_1gram']} unigrams, "
            f"{record['lm']['n_2gram']} bigrams, {record['lm']['n_3gram']} trigrams, "
            f"{record['lm']['n_states']} CSR states, binary {record['lm']['binary_bytes']} bytes",
            f"training text: {text['n_lines']} lines, {text['n_tokens']} tokens, "
            f"{text['n_types']} types",
            f"distinct bigram types: in_line={text['in_line']}, with_bos={text['with_bos']}, "
            f"with_bos_eos={text['with_bos_eos']} (banked {self.expected_bigram_types}, "
            f"matched by {', '.join(matched)})",
            "",
            f"null (seed {self.derangement_seed}, {null_stats['algorithm']}): "
            f"{null_stats['fixed_points']} fixed points, "
            f"{null_stats['kept_by_homophony']} words keep a HOMOPHONE of their own pronunciation; "
            f"child / word_start / is_word_end are bit-identical to the real trie",
            f"runtime {record['runtime_seconds']:.1f} s",
        ]
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        print("\n".join(lines), flush=True)


def get_lexicon_trie() -> LexiconTrieBuildJob:
    """The in-house resource (``LexiconTrieBuildJob.rlMsnTBSZXsB``'s arguments), from public inputs.

    The bliss + g2p lexicon and the LM text of :func:`~.lexicon.lm_corpus_lexicon_and_g2p`, the
    prior's own window (:func:`~.phone_text.get_prior_window`) and prior
    (:func:`~.phone_prior.get_phone_prior`), and every other argument at its default.  The window
    replay and the word trigram run as the jobs before it (module docstring): the replay with the
    source job's window defaults, ``KenLMplzJob`` with the source's ``lmplz`` arguments
    (:data:`WORD_LM_ORDER`, ``prior_gap.DISCOUNT_FALLBACK``, ``-S`` :data:`LMPLZ_MEM_GB` G).
    """
    from i6_core.lm.kenlm import CreateBinaryLMJob, KenLMplzJob

    from ..default_tools import get_kenlm_binary_path
    from .lexicon import lm_corpus_lexicon_and_g2p
    from .phone_prior import get_phone_prior
    from .phone_text import get_prior_window
    from .prior_gap import DISCOUNT_FALLBACK
    from .word_window import WordWindowReplayJob

    text, lex, g2p = lm_corpus_lexicon_and_g2p()
    window = WordWindowReplayJob(bliss_lexicon=lex, g2p_lexicon=g2p, word_corpus=text,
                                 window_phn=get_prior_window())
    window.add_alias("sae/4a/lexlat/resource_window")
    kenlm = get_kenlm_binary_path()
    lmplz = KenLMplzJob(
        text=[window.out_words],
        order=WORD_LM_ORDER,
        interpolate_unigrams=True,
        pruning=None,
        vocabulary=None,
        discount_fallback=list(DISCOUNT_FALLBACK),
        kenlm_binary_folder=kenlm,
        mem=LMPLZ_MEM_GB,
    )
    lmplz.add_alias("sae/4a/lexlat/resource_lmplz")
    binary = CreateBinaryLMJob(arpa_lm=lmplz.out_lm, kenlm_binary_folder=kenlm)
    binary.add_alias("sae/4a/lexlat/resource_build_binary")
    job = LexiconTrieBuildJob(
        bliss_lexicon=lex,
        g2p_lexicon=g2p,
        window_words=window.out_words,
        window_stats=window.out_stats,
        word_lm_arpa=lmplz.out_lm,
        word_lm_binary=binary.out_lm,
        prior_npz=get_phone_prior(),
        word_lm_order=WORD_LM_ORDER,
    )
    job.add_alias("sae/4a/lexlat/resource")
    return job
