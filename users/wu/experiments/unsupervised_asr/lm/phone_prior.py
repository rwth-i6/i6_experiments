"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/prior.py (the fit side and its job).

The phone m-gram P_psi(y) of SAE 4a: an interpolated Witten-Bell trigram over the 39 ARPAbet + SIL
inventory (trigram -> bigram -> unigram -> uniform), counted on the SIL-augmented phone corpus
(``phone_text.PhonemizeWithSilJob``, ``<SIL>`` at word boundaries with p_sil = 0.5) or, in every
banked arm, on its seeded uniform window (``phone_text.SampleLinesJob``, 1,010,000 lines, seed 0).
Witten-Bell is parameter-free, and its backoff weight T(h)/(N(h)+T(h)) is the standard choice when
the context inventory is tiny and dense.

``prior.npz`` keeps the source's format (keys ``log_uni``, ``log_bi``, ``log_tri``, ``phones``,
``meta``, ``tri_counts``), which the RETURNN side (``model/prior.py``) reads.

``PhoneNgramPrior``, ``NgramCounts`` and ``_witten_bell`` are not redefined here: they are imported
from ``model/prior.py`` (one copy of the table construction and its scorer). Only the fit side
(``read_phone_lines``, ``count_ngrams``, ``fit_prior``, ``PhoneNgramPriorJob``) lives in this module.
"""

from __future__ import annotations

import gzip
import json
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    from sisyphus import Job, Task, tk
except ImportError:  # standalone import of the pure helpers (tests, no sisyphus)
    Job, Task, tk = object, None, None

from ..phones import PHONES, SIL
from ..model.prior import NgramCounts, PhoneNgramPrior

__all__ = [
    "PHONE2ID",
    "SIL_ID",
    "BOS_ID",
    "N_TYPES",
    "N_CTX",
    "DEFAULT_COUNT_LINES",
    "DEFAULT_HELD_LINES",
    "HELD_STRIDE",
    "PhoneNgramPrior",
    "NgramCounts",
    "count_ngrams",
    "fit_prior",
    "read_phone_lines",
    "PhoneNgramPriorJob",
    "get_phone_prior",
]

PHONE2ID: Dict[str, int] = {p: i for i, p in enumerate(PHONES)}
SIL_ID = PHONE2ID[SIL]
N_TYPES = len(PHONES)  # 40 = 39 ARPAbet + SIL
BOS_ID = N_TYPES  # sentence-start context only; never a predicted symbol
N_CTX = N_TYPES + 1

# The SIL-augmented corpus spells silence ``<SIL>``; every other read in the campaign spells it
# ``SIL``. One alias table, applied on read.
_TOKEN_ALIASES = {"<SIL>": SIL, "[SIL]": SIL, "sil": SIL}

# Corpus budget for the count pass. A 40-symbol trigram has 67,240 cells, so 1 M lines (~81 M
# tokens) leaves ~1.2e3 observations per cell.
DEFAULT_COUNT_LINES = 1_000_000
DEFAULT_HELD_LINES = 10_000
HELD_STRIDE = 101  # every 101st line of the read window is held out instead of counted


def read_phone_lines(path: str, limit: Optional[int] = None) -> Iterable[List[str]]:
    """Yield the phone-token list of each line of a (optionally gzipped) phone corpus."""
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as fh:
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            yield line.split()


def _to_ids(tokens: Sequence[str]) -> List[int]:
    get = PHONE2ID.get
    out = []
    for t in tokens:
        i = get(t)
        if i is None:
            i = PHONE2ID.get(_TOKEN_ALIASES.get(t, ""), None)
            if i is None:
                raise KeyError(f"phone token {t!r} is outside the 39 ARPAbet + SIL inventory")
        out.append(i)
    return out


def count_ngrams(
    lines: Iterable[Sequence[str]], *, chunk: int = 50_000, held_stride: int = 0
) -> Tuple[NgramCounts, List[List[int]]]:
    """Count over ``lines``; every ``held_stride``-th line is withheld and returned instead."""
    counts = NgramCounts.zeros()
    buf: List[List[int]] = []
    held: List[List[int]] = []
    for i, toks in enumerate(lines):
        if not toks:
            continue
        ids = _to_ids(toks)
        if held_stride and i % held_stride == 0:
            held.append(ids)
            continue
        buf.append(ids)
        if len(buf) >= chunk:
            counts.add_lines(buf)
            buf = []
    counts.add_lines(buf)
    return counts, held


def fit_prior(
    corpus_path: str,
    *,
    n_count_lines: int = DEFAULT_COUNT_LINES,
    n_held_lines: int = DEFAULT_HELD_LINES,
    held_stride: int = HELD_STRIDE,
) -> Tuple[PhoneNgramPrior, dict]:
    """Count, smooth, and score the held-out lines. Returns ``(prior, stats)``."""
    n_read = n_count_lines + n_held_lines
    counts, held = count_ngrams(read_phone_lines(corpus_path, limit=n_read), held_stride=held_stride)
    held = held[:n_held_lines]
    meta = {
        "corpus": str(corpus_path),
        "lines_read": n_read,
        "lines_counted": counts.n_lines,
        "tokens_counted": counts.n_tokens,
        "lines_held": len(held),
        "held_stride": held_stride,
        "smoothing": "interpolated Witten-Bell",
        "inventory": "39 ARPAbet + SIL",
    }
    prior = PhoneNgramPrior.from_counts(counts, meta=meta)
    stats = dict(meta)
    for order in (1, 2, 3):
        stats[f"held_ppl_order{order}"] = prior.perplexity(held, order)
        stats[f"held_bits_per_phone_order{order}"] = math.log2(stats[f"held_ppl_order{order}"])
    stats["sil_token_rate_counted"] = float(counts.unigram[SIL_ID]) / max(counts.n_tokens, 1)
    return prior, stats


def _json_safe(o):
    """Recursively replace non-finite floats with ``None`` (speech-llm ``sae/json_io.json_safe``)."""
    if isinstance(o, float):
        return None if not math.isfinite(o) else o
    if isinstance(o, dict):
        return {k: _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    return o


def _dump_json(obj, path: str, **kw) -> None:
    """``json.dump`` with undefined values as ``null`` (speech-llm ``sae/json_io.dump_json``)."""
    with open(path, "w") as fh:
        json.dump(_json_safe(obj), fh, allow_nan=False, **kw)


# ---------------------------------------------------------------------------------------------------
# sisyphus job
# ---------------------------------------------------------------------------------------------------


class PhoneNgramPriorJob(Job):
    """Fit P_psi on a phone corpus and persist its tables + held-out perplexities.

    CPU, in-process (the compute is a gzip read and two ``bincount`` passes). The corpus is consumed
    as a frozen path; this job never rebuilds the phonemization.
    """

    def __init__(
        self,
        *,
        corpus_phn: tk.Path,
        n_count_lines: int = DEFAULT_COUNT_LINES,
        n_held_lines: int = DEFAULT_HELD_LINES,
    ):
        self.corpus_phn = corpus_phn
        self.n_count_lines = int(n_count_lines)
        self.n_held_lines = int(n_held_lines)
        self.out_prior = self.output_path("prior.npz")
        self.out_stats = self.output_path("prior.stats.txt")
        self.out_json = self.output_path("prior.json")
        self.rqmt = {"cpu": 2, "mem": 16, "time": 2}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        prior, stats = fit_prior(
            self.corpus_phn.get_path(),
            n_count_lines=self.n_count_lines,
            n_held_lines=self.n_held_lines,
        )
        prior.save(self.out_prior.get_path())
        lines = [f"phone m-gram prior P_psi <- {self.corpus_phn.get_path()}"] + [
            f"{k} = {v}" for k, v in stats.items()
        ]
        with open(self.out_stats.get_path(), "w") as fh:
            fh.write("\n".join(lines) + "\n")
        _dump_json(stats, self.out_json.get_path(), indent=2)
        print("\n".join(lines), flush=True)


def get_phone_prior() -> "tk.Path":
    """The phase's ``prior.npz`` (``PhoneNgramPriorJob.RtzbESkOedsT``'s arguments): the Witten-Bell
    trigram fitted on :func:`~.phone_text.get_prior_window` at the default count / held split."""
    from .phone_text import get_prior_window

    job = PhoneNgramPriorJob(corpus_phn=get_prior_window())
    job.add_alias("sae/4a/lm/prior")
    return job.out_prior
