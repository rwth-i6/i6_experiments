"""Ported from speech-llm c49559ce src/speech_llm/sae/emc/prior.py.

Port note: the RETURNN side only -- ``PhoneNgramPrior`` (load / save / score / class trigram),
``NgramCounts``, ``_witten_bell``, the manner-8 class partition.  The phone inventory comes from
``..phones`` (same order, same ids).  The fit (``read_phone_lines``, ``count_ngrams``,
``fit_prior``, ``PhoneNgramPriorJob``) lives in ``lm/phone_prior.py``.  The npz format is
unchanged.

emc.prior -- the phone m-gram P_psi(y) of SAE §4a (SAE_4A.md:72-74).

Text side: T_phi, the boundary-free phonemization of the LibriSpeech LM corpus
(``TextToPhonemeJob.THKMON3k9LJQ``, 39 ARPAbet + SIL), with SIL inserted at word boundaries with
p_sil = 0.5 -- the §1c convention, so the GAN comparison shares its text side.

T_phi's own output carries NO word boundaries (``phon_lm.py:222-236`` writes ``" ".join(phons)``, one
flat phone sequence per line), so SIL cannot be inserted into it after the fact. The corpus consumed
here is therefore the §1c phonemization WITH SIL of the same corpus, by the same lexicon, in the same
line order: ``PhonemizeWithSilJob.DbFgvZOGZQ8F`` (``sil_prob=0.5``, ``surround=True``, ``seed=0``,
40,418,261 -> 39,630,169 lines, identical to T_phi's line count), which is exactly the file
``FairseqPreprocessTextJob.bi2B89fES77z`` binarized for the §1c GAN. The insertion rule it applies is
``text.py:38-55``: one ``<SIL>`` before the first word and after the last (``surround``), and one
between word i and i+1 whenever ``draws[i] < sil_prob`` with one ``random_sample(W-1)`` draw per line.
Consuming that file rather than re-drawing SIL keeps the two text sides byte-identical.

Smoothing: **interpolated Witten-Bell** (trigram -> bigram -> unigram -> uniform). It is
parameter-free -- no smoothing constant has to be invented for a 40-symbol inventory -- and its
backoff weight T(h)/(N(h)+T(h)) is the standard choice when the context inventory is tiny and dense.

The gate reports ``log P_psi`` BESIDE the reverse term, never inside it (G4a.1, SAE_4A.md:93-96).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..phones import ARPABET_39, BOS_ID, N_CTX, N_TYPES, PHONE2ID, PHONES, SIL, SIL_ID

__all__ = [
    "ARPABET_39",
    "SIL",
    "PHONES",
    "SIL_ID",
    "BOS_ID",
    "N_TYPES",
    "N_CTX",
    "MANNER8",
    "manner8_class_ids",
    "class_trigram_log_table",
    "PhoneNgramPrior",
    "NgramCounts",
]

# The hand manner/place partition named in reports/estimate_prior_order_2026-09-15.full.md (D2:
# "vowel, stop, fricative, affricate, nasal, liquid, glide, SIL is a natural 8"), VERBATIM as the
# ``fixed8`` row of ``analysis/prior_order_ppl.py:87-97`` -- the free held-out read the class
# trigram was costed on, so the lattice's table and that perplexity are the same model.
MANNER8: Dict[str, List[str]] = {
    "vowel": ["AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"],
    "stop": ["B", "D", "G", "K", "P", "T"],
    "fricative": ["DH", "F", "HH", "S", "SH", "TH", "V", "Z", "ZH"],
    "affricate": ["CH", "JH"],
    "nasal": ["M", "N", "NG"],
    "liquid": ["L", "R"],
    "glide": ["W", "Y"],
    "sil": ["SIL"],
}


def manner8_class_ids() -> List[int]:
    """``[N_CTX]`` class id per CONTEXT symbol for :data:`MANNER8`; BOS gets its own class.

    BOS is a class of its own (id 8, so 9 classes and a lattice history axis of 9 x 41), which is
    the convention the held-out class-trigram perplexity was measured under
    (``analysis/prior_order_ppl.py:220,264,275,377``); folding BOS into a phone class would score a
    different model than the one that was read. The class of a symbol is its manner, so the class
    trigram conditions on ``(manner(p_-2), p_-1)``.
    """
    out = [-1] * N_CTX
    for ci, name in enumerate(MANNER8):
        for ph in MANNER8[name]:
            out[PHONE2ID[ph]] = ci
    assert all(c >= 0 for c in out[:N_TYPES]), "the manner partition misses a phone"
    out[BOS_ID] = len(MANNER8)
    return out


def _witten_bell(c, backoff):
    """Interpolated Witten-Bell ``p(w|h) = (c(h,w) + T(h) p(w|h')) / (N(h) + T(h))``.

    An unseen context falls back to ``backoff`` outright. This is the ONE implementation: the
    bigram, the trigram and the class trigram are all built through it.
    """
    import numpy as np

    n = c.sum(axis=-1, keepdims=True)
    t = (c > 0).sum(axis=-1, keepdims=True).astype(np.float64)
    out = (c + t * backoff) / np.maximum(n + t, 1.0)
    return np.where(n > 0, out, backoff * np.ones_like(out))


def class_trigram_log_table(trigram_counts, classes, log_bi):
    """``[n_cls * N_CTX, N_TYPES]`` ``log p(w | class(h2), h1)``, row ``class(h2) * N_CTX + h1``.

    A MARGINALISATION of the already-fitted trigram counts, not a second fit: the counts of the
    ``(h2, h1)`` rows are summed inside each class of ``h2`` and the same interpolated Witten-Bell
    (:func:`_witten_bell`) then backs the class row off to the bigram ``log_bi`` of the same fit --
    exactly the estimator whose held-out perplexity ``analysis/prior_order_ppl.py`` reports.

    :param trigram_counts: ``NgramCounts.trigram``, ``[N_CTX * N_CTX, N_TYPES]`` raw counts.
    :param classes: ``[N_CTX]`` class id per context symbol (see :func:`manner8_class_ids`).
    :param log_bi: ``[N_CTX, N_TYPES]`` the fit's own bigram table, the backoff distribution.

    With ``classes = [0, 1, ..., N_CTX - 1]`` (every symbol its own class) the result is the full
    trigram table of the same fit, which is what the test asserts.
    """
    import numpy as np

    tri = np.asarray(trigram_counts, dtype=np.float64)
    assert tri.shape == (N_CTX * N_CTX, N_TYPES), tri.shape
    cmap = np.asarray(list(classes), dtype=np.int64)
    assert cmap.shape == (N_CTX,), (
        f"classes needs one class id per context symbol ({N_CTX}, BOS last), got {cmap.shape}"
    )
    assert int(cmap.min()) >= 0, "class ids are 0-based and contiguous"
    n_cls = int(cmap.max()) + 1
    p2 = np.exp(np.asarray(log_bi, dtype=np.float64))
    assert p2.shape == (N_CTX, N_TYPES), p2.shape
    # row r of the trigram table is (h2 = r // N_CTX, h1 = r % N_CTX): sum over the h2 of a class
    acc = np.zeros((n_cls, N_CTX, N_TYPES), dtype=np.float64)
    np.add.at(acc, cmap, tri.reshape(N_CTX, N_CTX, N_TYPES))
    # np.tile, not np.repeat: row (c, h1) backs off to p2[h1] (the same rule as the trigram's)
    with np.errstate(divide="ignore"):
        return np.log(_witten_bell(acc.reshape(n_cls * N_CTX, N_TYPES), np.tile(p2, (n_cls, 1))))


@dataclass
class NgramCounts:
    """Raw counts; the tables are built from these so the fit is auditable."""

    unigram: "object"  # np.ndarray [N_TYPES]
    bigram: "object"  # np.ndarray [N_CTX, N_TYPES]
    trigram: "object"  # np.ndarray [N_CTX * N_CTX, N_TYPES]
    n_lines: int = 0
    n_tokens: int = 0

    @staticmethod
    def zeros() -> "NgramCounts":
        import numpy as np

        return NgramCounts(
            unigram=np.zeros(N_TYPES, dtype=np.int64),
            bigram=np.zeros((N_CTX, N_TYPES), dtype=np.int64),
            trigram=np.zeros((N_CTX * N_CTX, N_TYPES), dtype=np.int64),
        )

    def add_lines(self, lines: Sequence[Sequence[int]]) -> None:
        """Accumulate one chunk. Each line is padded with two BOS, so no context crosses a line."""
        import numpy as np

        if not lines:
            return
        stream: List[int] = []
        for ids in lines:
            stream.append(BOS_ID)
            stream.append(BOS_ID)
            stream.extend(ids)
            self.n_lines += 1
            self.n_tokens += len(ids)
        arr = np.asarray(stream, dtype=np.int64)
        pos = np.flatnonzero(arr != BOS_ID)
        pos = pos[pos >= 2]
        if pos.size == 0:
            return
        w = arr[pos]
        h1 = arr[pos - 1]
        h2 = arr[pos - 2]
        self.unigram += np.bincount(w, minlength=N_TYPES)
        self.bigram += np.bincount(h1 * N_TYPES + w, minlength=N_CTX * N_TYPES).reshape(N_CTX, N_TYPES)
        self.trigram += np.bincount(
            (h2 * N_CTX + h1) * N_TYPES + w, minlength=N_CTX * N_CTX * N_TYPES
        ).reshape(N_CTX * N_CTX, N_TYPES)


class PhoneNgramPrior:
    """log P_psi(y) at order 1, 2 or 3 over the 39 ARPAbet + SIL inventory.

    Convention (pre-registered here, with the producing code -- a rule that lives only in a plan file
    is discharged by whoever reads the number first): ``log_prob`` is the sum of per-token
    conditional log-probabilities with the context padded by sentence-start symbols and **no
    end-of-sequence term**, so the value is comparable across the matched-length conditions of S1a.
    """

    def __init__(self, log_uni, log_bi, log_tri, *, meta: Optional[dict] = None, tri_counts=None):
        import numpy as np

        self.log_uni = np.asarray(log_uni, dtype=np.float64)
        self.log_bi = np.asarray(log_bi, dtype=np.float64)
        self.log_tri = np.asarray(log_tri, dtype=np.float64)
        assert self.log_uni.shape == (N_TYPES,)
        assert self.log_bi.shape == (N_CTX, N_TYPES)
        assert self.log_tri.shape == (N_CTX * N_CTX, N_TYPES)
        # The RAW trigram counts of the same fit, kept so a class-based history can be marginalised
        # out of them later (:func:`class_trigram_log_table`). Optional: a prior.npz written before
        # they were persisted loads without them, and only the class trigram misses them.
        self.tri_counts = None if tri_counts is None else np.asarray(tri_counts, dtype=np.int64)
        if self.tri_counts is not None:
            assert self.tri_counts.shape == (N_CTX * N_CTX, N_TYPES), self.tri_counts.shape
        self.meta = dict(meta or {})

    # -- construction -------------------------------------------------------------------------------

    @classmethod
    def from_counts(cls, counts: NgramCounts, *, meta: Optional[dict] = None) -> "PhoneNgramPrior":
        """Interpolated Witten-Bell: p(w|h) = (c(h,w) + T(h) p(w|h')) / (N(h) + T(h))."""
        import numpy as np

        uni = counts.unigram.astype(np.float64)
        n0, t0 = uni.sum(), float((uni > 0).sum())
        p_uniform = np.full(N_TYPES, 1.0 / N_TYPES)
        p1 = (uni + t0 * p_uniform) / max(n0 + t0, 1.0) if n0 > 0 else p_uniform
        _wb = _witten_bell
        p2 = _wb(counts.bigram.astype(np.float64), p1.reshape(1, -1))
        # row r of the trigram table is (h2 = r // N_CTX, h1 = r % N_CTX), so its backoff is
        # p2[h1] -- np.tile, not np.repeat (which would back off on the WRONG context).
        p3 = _wb(counts.trigram.astype(np.float64), np.tile(p2, (N_CTX, 1)))
        with np.errstate(divide="ignore"):
            return cls(np.log(p1), np.log(p2), np.log(p3), meta=meta, tri_counts=counts.trigram)

    # -- scoring ------------------------------------------------------------------------------------

    def per_token_log_probs(self, seq: Sequence[int], order: int = 2):
        """``[len(seq)]`` conditional log-probabilities under the requested order."""
        import numpy as np

        assert order in (1, 2, 3), order
        ids = np.asarray(list(seq), dtype=np.int64)
        if ids.size == 0:
            return np.zeros(0)
        if order == 1:
            return self.log_uni[ids]
        h1 = np.concatenate([[BOS_ID], ids[:-1]])
        if order == 2:
            return self.log_bi[h1, ids]
        h2 = np.concatenate([[BOS_ID, BOS_ID], ids[:-2]])
        return self.log_tri[h2 * N_CTX + h1, ids]

    def log_prob(self, seq: Sequence[int], order: int = 2) -> float:
        return float(self.per_token_log_probs(seq, order).sum())

    def class_trigram(self, classes):
        """``log p(w | class(h2), h1)`` from THIS fit's trigram counts (:func:`class_trigram_log_table`)."""
        if self.tri_counts is None:
            raise ValueError(
                "this prior.npz carries no raw trigram counts, so a class trigram cannot be "
                "marginalised out of it; re-run PhoneNgramPriorJob (same hash, same tables, it "
                "writes 'tri_counts' into prior.npz as well)"
            )
        return class_trigram_log_table(self.tri_counts, classes, self.log_bi)

    def unigram_probs(self):
        """The unigram table S1a's unigram-draw null samples from."""
        import numpy as np

        p = np.exp(self.log_uni)
        return p / p.sum()

    def perplexity(self, sequences: Iterable[Sequence[int]], order: int = 2) -> float:
        total, n = 0.0, 0
        for s in sequences:
            total += self.log_prob(s, order)
            n += len(s)
        return float(math.exp(-total / max(n, 1)))

    # -- io -----------------------------------------------------------------------------------------

    def save(self, path: str) -> None:
        import numpy as np

        arrays = dict(
            log_uni=self.log_uni, log_bi=self.log_bi, log_tri=self.log_tri,
            phones=np.array(PHONES), meta=np.array([repr(self.meta)]),
        )
        if self.tri_counts is not None:
            # The raw counts of the same fit, so a class-based history can be marginalised out of
            # the banked table without a second corpus pass. Additive: the tables above are
            # byte-for-byte what they were, and ``load`` tolerates their absence.
            arrays["tri_counts"] = self.tri_counts
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "PhoneNgramPrior":
        import ast

        import numpy as np

        d = np.load(path, allow_pickle=False)
        assert list(d["phones"]) == PHONES, "the stored prior uses a different phone inventory"
        meta = ast.literal_eval(str(d["meta"][0])) if "meta" in d else {}
        counts = d["tri_counts"] if "tri_counts" in d.files else None
        return cls(d["log_uni"], d["log_bi"], d["log_tri"], meta=meta, tri_counts=counts)
