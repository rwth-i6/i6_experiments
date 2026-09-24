"""Ported from i6_experiments 5207c8adf users/wu/experiments/unsupervised_asr/w2vu2/text.py
(``SIL``, ``phonemize_line``, ``PhonemizeWithSilJob``), speech-llm c49559ce
src/speech_llm/sae/emc/text_sample.py (``SampleLinesJob`` and helpers).  ``ExcludeLinesJob``
(src/speech_llm/sae/emc/text_filter.py) is not ported: no phase-4a run uses it.

The phone text of phase 4a:

* :class:`PhonemizeWithSilJob` -- the LibriSpeech LM corpus as ARPAbet lines with the wav2vec-U
  silence protocol (fairseq ``scripts/phonemize_with_sil.py``): ``<SIL>`` around every sentence
  (``surround``) and after word i (internal boundaries only) with probability ``sil_prob``, one
  ``random_sample(W-1)`` draw per kept line; lines with an OOV word are dropped. One deterministic
  pronunciation per word: the bliss lexicon first, the g2p lexicon for the rest.
* :class:`SampleLinesJob` -- a seeded UNIFORM line sample of that corpus. The LM corpus is
  alphabetically sorted, so the head window the prior used to read is biased; the banked arms fit
  the prior on ``SampleLinesJob(n_out=1,010,000, seed=0)``. The sample is a function of
  (line count, ``n_out``, ``seed``) alone.

Cut from the w2vu2 source: ``W2VU_PYTHON`` and ``assert_w2vu_env`` (fairseq env, GAN only).
"""

from __future__ import annotations

import gzip
import math
import random
from collections import Counter
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

try:
    from sisyphus import Job, Task, tk
except ImportError:  # standalone import of the pure helpers (tests, no sisyphus)
    Job, Task, tk = object, None, None

__all__ = [
    "SIL",
    "phonemize_line",
    "PhonemizeWithSilJob",
    "SIL_TOKEN",
    "count_lines",
    "sample_line_indices",
    "iter_sampled",
    "initial_phone_counts",
    "format_stats",
    "SampleLinesJob",
    "get_phone_corpus",
    "get_prior_window",
]


# ===================================================================================================
# the SIL-augmented phonemization (w2vu2/text.py)
# ===================================================================================================
SIL = "<SIL>"


def phonemize_line(words, word_to_phon, sil_prob, surround, rng):
    """One line -> phone tokens, or None if any word is OOV.

    Draw order and count mirror fairseq `phonemize_with_sil.py` exactly (one `random_sample(W-1)`
    per kept line, consumed left to right), so a shared seed reproduces its output token for token.
    """
    prons = [word_to_phon.get(w) for w in words]
    if not words or any(p is None for p in prons):
        return None
    phones = [SIL] if surround else []
    draws = rng.random_sample(len(words) - 1) if sil_prob > 0 and len(words) > 1 else None
    for i, pron in enumerate(prons):
        phones.extend(pron.split())
        if draws is not None and i < len(draws) and draws[i] < sil_prob:
            phones.append(SIL)
    if surround:
        phones.append(SIL)
    return phones


class PhonemizeWithSilJob(Job):
    """Text -> ARPAbet lines with the wav2vec-U silence protocol. One deterministic pron per word."""

    def __init__(
        self,
        *,
        text_file: tk.Path,
        bliss_lexicon: tk.Path,
        g2p_lexicon: Optional[tk.Path] = None,
        sil_prob: float = 0.5,
        surround: bool = True,
        seed: int = 0,
        max_lines: Optional[int] = None,
    ):
        super().__init__()
        self.text_file = text_file
        self.bliss_lexicon = bliss_lexicon
        self.g2p_lexicon = g2p_lexicon
        self.sil_prob = sil_prob
        self.surround = surround
        self.seed = seed
        self.max_lines = max_lines

        self.out_text = self.output_path("text.phn.gz")
        self.out_stats = self.output_path("stats.txt")
        self.rqmt = {"cpu": 1, "mem": 8, "time": 6}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        import numpy as np

        from i6_core.util import uopen

        from .lexicon import _load_g2p_lexicon, _load_lexicon_word_to_phon

        w2p = _load_lexicon_word_to_phon(self.bliss_lexicon.get_path())
        if self.g2p_lexicon is not None:
            for w, p in _load_g2p_lexicon(self.g2p_lexicon.get_path()).items():
                w2p.setdefault(w, p)

        rng = np.random.RandomState(self.seed)
        n_in = n_out = n_sil = n_tok = 0
        with uopen(self.text_file.get_path(), "rt") as inf, uopen(self.out_text.get_path(), "wt") as outf:
            for line in inf:
                if self.max_lines is not None and n_in >= self.max_lines:
                    break
                n_in += 1
                phones = phonemize_line(line.split(), w2p, self.sil_prob, self.surround, rng)
                if phones is None:
                    continue

                outf.write(" ".join(phones) + "\n")
                n_out += 1
                n_tok += len(phones)
                n_sil += sum(1 for p in phones if p == SIL)

        with uopen(self.out_stats.get_path(), "wt") as f:
            f.write(
                f"lines_in={n_in}\nlines_out={n_out}\ndropped_oov={n_in - n_out}\n"
                f"sil_prob={self.sil_prob}\nsurround={self.surround}\nseed={self.seed}\n"
                f"tokens={n_tok}\nsil_tokens={n_sil}\nsil_token_rate={n_sil / max(n_tok, 1):.6f}\n"
            )


# ===================================================================================================
# the seeded uniform line sample (sae/emc/text_sample.py)
# ===================================================================================================
# The sentence-initial phone of a line is the token AFTER the leading ``<SIL>`` (``surround=True``).
SIL_TOKEN = "<SIL>"


def _opener(path: str):
    return gzip.open if str(path).endswith(".gz") else open


def count_lines(path: str) -> int:
    """Number of lines of a (optionally gzipped) text file, counted on the raw bytes."""
    opener = _opener(path)
    total = 0
    with opener(path, "rb") as fh:
        while True:
            chunk = fh.read(1 << 24)
            if not chunk:
                break
            total += chunk.count(b"\n")
    return total


def sample_line_indices(n_lines: int, n_out: int, seed: int) -> List[int]:
    """The SORTED indices of a seeded uniform random subset of ``n_out`` of ``n_lines`` lines.

    Sorted, so the selected lines are written in the corpus's own relative order -- the sample is a
    subset of the corpus, not a shuffle of it.
    """
    n_lines, n_out, seed = int(n_lines), int(n_out), int(seed)
    assert 0 <= n_out <= n_lines, f"cannot draw {n_out} of {n_lines} lines"
    return sorted(random.Random(seed).sample(range(n_lines), n_out))


def iter_sampled(lines: Iterable[str], keep: Sequence[int]) -> Iterator[str]:
    """Yield the lines at the SORTED indices ``keep``, streaming, in the input's own order."""
    keep = list(keep)
    assert all(a < b for a, b in zip(keep, keep[1:])), "keep must be sorted and duplicate-free"
    wanted = iter(keep)
    nxt = next(wanted, None)
    for i, line in enumerate(lines):
        if nxt is None:
            return
        if i == nxt:
            yield line
            nxt = next(wanted, None)


def initial_phone_counts(lines: Iterable[str]) -> Tuple[Counter, Dict[str, int]]:
    """``(Counter of the sentence-initial phone, per-line tallies)`` over ``lines``.

    The sentence-initial phone is the token AFTER the leading :data:`SIL_TOKEN`.  A line that does
    not start with ``<SIL>`` is counted in ``n_no_leading_sil`` and contributes nothing to the
    Counter; a line with no token after the leading ``<SIL>`` is counted in ``n_no_initial_phone``.
    """
    counts: Counter = Counter()
    tally = {"n_lines": 0, "n_no_leading_sil": 0, "n_no_initial_phone": 0}
    for line in lines:
        tokens = line.split()
        tally["n_lines"] += 1
        if not tokens or tokens[0] != SIL_TOKEN:
            tally["n_no_leading_sil"] += 1
            continue
        if len(tokens) < 2:
            tally["n_no_initial_phone"] += 1
            continue
        counts[tokens[1]] += 1
    return counts, tally


def format_stats(*, corpus: str, n_in: int, n_out: int, seed: int, counts: Counter,
                 tally: Dict[str, int], top: int = 10) -> str:
    """The ``stats.txt`` body: the draw's parameters and the sample's sentence-initial phones."""
    lines = [
        f"seeded uniform line sample <- {corpus}",
        f"n_in = {n_in}",
        f"n_out = {n_out}",
        f"seed = {seed}",
        f"lines_written = {tally['n_lines']}",
        f"n_no_leading_sil = {tally['n_no_leading_sil']}",
        f"n_no_initial_phone = {tally['n_no_initial_phone']}",
        f"initial phone (the token after the leading {SIL_TOKEN}) of the SAMPLED lines, "
        f"top {top} of {len(counts)} types:",
    ]
    total = max(sum(counts.values()), 1)
    for phone, n in counts.most_common(top):
        lines.append(f"  {phone} = {n} ({n / total:.4f})")
    return "\n".join(lines) + "\n"


class SampleLinesJob(Job):
    """Write a seeded uniform random subset of ``n_out`` lines of a text corpus, order preserved.

    CPU, in-process: two gzip reads (one to count the lines, one to select them) and one gzip
    write.  The corpus is consumed as a frozen path; this job never rebuilds the phonemization.
    """

    def __init__(self, *, corpus: tk.Path, n_out: int, seed: int = 0):
        self.corpus = corpus
        self.n_out = int(n_out)
        self.seed = int(seed)
        self.out_corpus = self.output_path("text.phn.gz")
        self.out_stats = self.output_path("stats.txt")
        self.rqmt = {"cpu": 2, "mem": 8, "time": 4}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        path = self.corpus.get_path()
        n_in = count_lines(path)
        keep = sample_line_indices(n_in, self.n_out, self.seed)
        opener = _opener(path)
        with opener(path, "rt") as src, gzip.open(self.out_corpus.get_path(), "wt") as dst:
            for line in iter_sampled(src, keep):
                dst.write(line if line.endswith("\n") else line + "\n")
        # The statistics are read back off the WRITTEN file, so they describe the sample this job
        # actually hands on rather than a second copy of the selection logic.
        with gzip.open(self.out_corpus.get_path(), "rt") as fh:
            counts, tally = initial_phone_counts(fh)
        assert tally["n_lines"] == self.n_out, (
            f"wrote {tally['n_lines']} lines, asked for {self.n_out} of {n_in}"
        )
        report = format_stats(corpus=path, n_in=n_in, n_out=self.n_out, seed=self.seed,
                              counts=counts, tally=tally)
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write(report)
        print(report, flush=True)


def _json_safe(o):
    """Recursively replace non-finite floats with ``None`` (speech-llm ``sae/json_io.json_safe``)."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


# ===================================================================================================
# the wiring: the phase's phone corpus and the prior's window
# ===================================================================================================
def get_phone_corpus() -> "tk.Path":
    """The SIL-augmented phone corpus (``PhonemizeWithSilJob.DbFgvZOGZQ8F``'s arguments).

    The LM text, bliss lexicon and g2p lexicon of :func:`~.lexicon.lm_corpus_lexicon_and_g2p`, and
    ``sil_prob = 0.5``, ``surround = True``, ``seed = 0``, all lines -- the call of the source's
    ``w2vu2/pipeline.py`` ``_text_data(sil_prob=0.5, max_lines=None)``.
    """
    from .lexicon import lm_corpus_lexicon_and_g2p

    text, lex, g2p = lm_corpus_lexicon_and_g2p()
    sil = PhonemizeWithSilJob(
        text_file=text, bliss_lexicon=lex, g2p_lexicon=g2p,
        sil_prob=0.5, surround=True, seed=0, max_lines=None,
    )
    sil.add_alias("sae/1c/text_sil0.5")
    return sil.out_text


def get_prior_window() -> "tk.Path":
    """The prior's window (``SampleLinesJob.orN768ARKwlt``'s arguments): 1,010,000 lines, seed 0.

    ``n_out`` is ``phone_prior.DEFAULT_COUNT_LINES + DEFAULT_HELD_LINES``, the source's own sum.
    """
    from .phone_prior import DEFAULT_COUNT_LINES, DEFAULT_HELD_LINES

    sample = SampleLinesJob(corpus=get_phone_corpus(),
                            n_out=DEFAULT_COUNT_LINES + DEFAULT_HELD_LINES, seed=0)
    sample.add_alias("sae/4a/lm/prior_sample")
    return sample.out_corpus
