"""SAE 4A attribution, step 1 -- training-free n-gram mode-seeking check (CPU, no model forward).

Reader for the diagnostic registered in ``SAE_4A_attrib.md`` ("Step 1"): does the cold cycle
recognizer place its mass on the *frequent* n-grams of the text side (mode seeking) while the GAN
covers the text distribution?  Nothing here is trained; the job re-fits one phone trigram on the
text corpus and scores banked decodes with it.

PRE-REGISTERED RULES (they live here, with the producing code, not only in the plan file):

Rows              blankfree ep1, blankfree ep4, GAN seed 0, gold -- all on the SAME dev-other
                  utterance set (the reference row's ids; every row must cover it exactly, so every
                  reported difference is paired over utterances).
Primary strings   SIL-REMOVED for every row and for the text side.  SIL tokens found in a primary
                  string are stripped and counted (``sil_stripped``); ``<SIL>``/``[SIL]``/``sil``
                  spellings are mapped onto ``SIL`` by ``emc.prior``'s one alias table before that,
                  so a fairseq-dictionary hypothesis and an EMC decode share the inventory.  Every
                  row is asserted to be inside the 39 ARPAbet monophones; the text side and
                  the reference (gold) row are asserted to realise all 39.  A row may also be
                  flagged ``require_full_inventory``, but the GAN row is NOT: the same checkpoint's
                  train-split dump realises 38 of the 39 (it never emits ZH), so the full-inventory
                  assert would abort the read (code review 2026-09-19); the subset assert stands
                  for every row.  A row file may cover more utterances than the reference set (the
                  GAN decode covers dev-clean + dev-other): it is RESTRICTED to the reference ids,
                  the surplus is counted in ``ids_dropped``, and no reference id may be missing, so
                  the restricted id set equals the reference set exactly.
Estimator         ONE trigram, fit by this job on the SIL-stripped text corpus and used for every
                  row: interpolated Witten-Bell (trigram -> bigram -> unigram -> uniform), i.e.
                  ``emc.prior.PhoneNgramPrior.from_counts``, the single smoothing implementation of
                  this campaign (parameter-free, so no smoothing constant has to be invented).  The
                  table keeps its 40-symbol axis; SIL is simply never counted, so it retains only
                  the uniform backoff mass (< 1.3e-8 per token) and no row can emit it.
Corpus window     the same lines the TRAINING prior was fit on (``PhoneNgramPriorJob``: the first
                  ``n_count_lines + n_held_lines`` lines of the corpus, every ``held_stride``-th
                  line withheld), so the text side of this read and of the trained prior differ
                  only by the SIL stripping.
log P             natural log (nats).  Per-token conditional log-probabilities with the context
                  padded by two sentence-start symbols and NO end-of-sequence term
                  (``PhoneNgramPrior``'s convention); "mean log-prob per phone" is the total over a
                  row divided by that row's phone count (token weighted, not utterance averaged).
JSD               base 2 (bits), n = 1..4, between the row's n-gram distribution POOLED over its
                  utterances and the text corpus's; n-grams are taken inside one utterance / one
                  corpus line only (no BOS/EOS padding, no context across a boundary).
                  JSD(P,Q) = 0.5 KL(P||M) + 0.5 KL(Q||M), M = (P+Q)/2.
Count matching    a plug-in JSD DEPENDS ON THE ROW'S TOKEN COUNT (fewer tokens -> sparser observed
                  support -> larger JSD; measured on gold dev-other by the code review of
                  2026-09-19: +0.013 bits at -15% of the tokens, +0.064 at -50%) and the rows
                  differ in phone count, so the PRIMARY read is COUNT MATCHED.  The budget is the
                  smallest phone count over the scored rows; for the point estimate and inside
                  EVERY bootstrap replicate each row is subsampled by WHOLE utterances -- one
                  shuffled utterance order, shared by all rows, truncated at the first utterance
                  that would push the row past the budget -- and every JSD and every
                  pre-registered difference is computed on that subsample.  Inside a replicate the
                  budget is lowered to the smallest resampled row total when a row cannot reach
                  the pre-registered budget, so the rows are always matched to each other.  The
                  shuffle is a fixed stream derived from ``seed``.  The unmatched full-count
                  statistics are kept and rendered as a SECONDARY table.
Secondary         mean log P3 per token under the TRAINING SIL-INCLUSIVE trigram (``prior_npz``) on
                  the SIL-inclusive strings, for the model rows that have one; gold is n/a.
Uncertainty       utterance-block bootstrap, ``n_bootstrap`` resamples, ``seed``.  The SAME
                  resampled utterance multiset is used for every row inside a replicate, so every
                  difference is paired.  The text side is fixed (not resampled): the intervals
                  describe utterance sampling of the rows only.  Interval conventions, fixed before
                  the read: a per-row mean log-prob carries the PERCENTILE CI95; a per-row JSD
                  carries the plug-in point estimate with the REVERSE-PERCENTILE (bias-corrected)
                  CI95, because a resample covers only ~63% of the distinct utterances, which
                  shrinks the observed n-gram support and inflates a plug-in JSD; every decisive
                  comparison is a per-resample difference summarised by the PERCENTILE CI95.  The
                  json carries ``ci95``, ``ci95_basic`` and ``bootstrap_bias`` for every statistic.
Rendered          ``summary.md`` MUST render, explicitly and each as a PAIRED-difference bootstrap
                  with its CI95 (SAE_4A_attrib.md "Design-review amendments", 2026-09-19):
                    (a) ep4 mean SIL-free trigram log-prob per phone MINUS the GAN's,
                    (b) ep4 4-gram JSD MINUS gold's,
                    (c) ep4 4-gram JSD MINUS the GAN's,
                    (d) GAN 4-gram JSD MINUS gold's.
                  Each carries its PRE-REGISTERED margin and direction and a PASS/FAIL cell
                  computed from the banked numbers (``PREREGISTERED_COMPARISONS`` below, from
                  SAE_4A_attrib.md's design-review amendments): (a) >= -0.10 nats; (b) and (c)
                  >= +0.05 bits AND the difference CI95 excluding zero; (d) within +-0.10 bits.
                  Both rows' phone counts, full and matched, are printed beside every comparison.
                  ``jsd4_reference_line`` (0.27) is Lin's corpus-size-dependent gold-vs-text value
                  and is rendered as a DESCRIPTIVE reference line beside the 4-gram JSDs only; it
                  decides nothing.  A comparison whose row is absent is rendered as "n/a" naming
                  the missing row -- never silently dropped.  Utterance count and phone count are
                  reported per row.
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from sisyphus import Job, Task, tk

from speech_llm.sae.emc import prior as _prior

__all__ = ["NgramModeSeekingJob", "PhoneRow", "ngram_counts", "jensen_shannon_bits",
           "trigram_log_probs", "analyse"]

BASE = _prior.N_TYPES  # 40 = 39 ARPAbet + SIL; the packing radix of an n-gram id
_SENTINEL = _prior.N_TYPES  # marks a line break in a packed stream (never a phone id)

# The step-1 predictions, fixed before the numbers (SAE_4A_attrib.md, "Design-review amendments",
# 2026-09-19).  They live here, with the producing code, so summary.md states what was predicted
# and whether it held; ``rule`` "ge" = value >= margin, "abs_le" = |value| <= margin.
PREREGISTERED_COMPARISONS = {
    "ep4_minus_gan_logprob": {
        "tag": "a", "rule": "ge", "margin": -0.10, "unit": "nats", "ci_excludes_zero": False,
        "direction": ">= -0.10 nats (cold ep4 log-prob at least the GAN's minus 0.10)"},
    "ep4_minus_gold_jsd4": {
        "tag": "b", "rule": "ge", "margin": 0.05, "unit": "bits", "ci_excludes_zero": True,
        "direction": ">= +0.05 bits and the difference CI95 excluding zero"},
    "ep4_minus_gan_jsd4": {
        "tag": "c", "rule": "ge", "margin": 0.05, "unit": "bits", "ci_excludes_zero": True,
        "direction": ">= +0.05 bits and the difference CI95 excluding zero"},
    "gan_minus_gold_jsd4": {
        "tag": "d", "rule": "abs_le", "margin": 0.10, "unit": "bits", "ci_excludes_zero": False,
        "direction": "within +-0.10 bits of gold"},
}


# ---------------------------------------------------------------------------------------------
# n-gram primitives
# ---------------------------------------------------------------------------------------------


def pack_ngrams(ids: Sequence[int], order: int) -> "np.ndarray":
    """Packed ids of the ``order``-grams fully inside ONE sequence (radix ``BASE``, no padding)."""
    arr = np.asarray(list(ids), dtype=np.int64)
    if arr.size < order:
        return np.zeros(0, dtype=np.int64)
    out = np.zeros(arr.size - order + 1, dtype=np.int64)
    for k in range(order):
        out = out * BASE + arr[k: arr.size - order + 1 + k]
    return out


def ngram_counts(lines: Sequence[Sequence[int]], order: int) -> "np.ndarray":
    """Dense ``[BASE ** order]`` counts of the ``order``-grams inside the given sequences."""
    if not lines:
        return np.zeros(BASE ** order, dtype=np.int64)
    stream: List[int] = []
    for ids in lines:
        stream.extend(ids)
        stream.append(_SENTINEL)  # windows crossing it are dropped below
    arr = np.asarray(stream, dtype=np.int64)
    ok = arr != _SENTINEL
    n = arr.size - order + 1
    if n <= 0:
        return np.zeros(BASE ** order, dtype=np.int64)
    idx = np.zeros(n, dtype=np.int64)
    valid = np.ones(n, dtype=bool)
    for k in range(order):
        idx = idx * BASE + arr[k: k + n]
        valid &= ok[k: k + n]
    return np.bincount(idx[valid], minlength=BASE ** order).astype(np.int64)


def jensen_shannon_bits(p: "np.ndarray", q_support: "np.ndarray", q_outside: float) -> float:
    """JSD in BITS between ``p`` (a distribution supported on the given support) and the reference.

    ``q_support`` is the reference's mass on the same support and ``q_outside`` its mass everywhere
    else.  Outside the support p is zero, so that region contributes 0.5 * q_outside exactly
    (q log2(q / (q/2)) = q), which is why only the row's own support has to be materialised.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q_support, dtype=np.float64)
    m = 0.5 * (p + q)
    out = 0.0
    sel = p > 0
    out += 0.5 * float(np.sum(p[sel] * np.log2(p[sel] / m[sel])))
    sel = q > 0
    out += 0.5 * float(np.sum(q[sel] * np.log2(q[sel] / m[sel])))
    out += 0.5 * max(float(q_outside), 0.0)
    return out


# ---------------------------------------------------------------------------------------------
# rows
# ---------------------------------------------------------------------------------------------


class PhoneRow:
    """One scored row: per-utterance phone id sequences, SIL-free (primary) and raw (secondary)."""

    def __init__(self, name: str, utt_ids: Sequence[str], primary: Sequence[Sequence[int]],
                 raw: Optional[Sequence[Sequence[int]]] = None, sil_stripped: int = 0,
                 ids_dropped: int = 0, secondary_na_reason: Optional[str] = None):
        self.name = name
        self.utt_ids = list(utt_ids)
        self.primary = [np.asarray(s, dtype=np.int64) for s in primary]
        self.raw = None if raw is None else [np.asarray(s, dtype=np.int64) for s in raw]
        self.sil_stripped = int(sil_stripped)
        # utterances present in the row's file but outside the reference set (e.g. the GAN decode
        # covers dev-clean + dev-other and is restricted here to the dev-other reference ids)
        self.ids_dropped = int(ids_dropped)
        self.secondary_na_reason = secondary_na_reason
        assert len(self.primary) == len(self.utt_ids)
        assert self.raw is None or len(self.raw) == len(self.utt_ids)

    @property
    def n_utts(self) -> int:
        return len(self.utt_ids)

    @property
    def n_phones(self) -> int:
        return int(sum(int(s.size) for s in self.primary))

    def inventory(self) -> List[str]:
        seen = set()
        for s in self.primary:
            seen.update(int(i) for i in s.tolist())
        return [_prior.PHONES[i] for i in sorted(seen)]


def load_phone_json(path: str, split: Optional[str] = None) -> Dict[str, List[str]]:
    """``{utt_id: [phone, ...]}`` from a decode json, tolerating the shapes banked in this setup.

    Accepted: ``{id: [phones]}``, ``{id: "P1 P2 ..."}``, ``{split: {id: ...}}`` (gold) and
    ``{"labels": {id: ...}, ...}`` (the fairseq pseudo-label dump).  ``split`` selects one split.
    """
    with open(path) as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "labels" in data and isinstance(data["labels"], dict):
        data = data["labels"]
    if split is not None:
        assert split in data, f"{path}: split {split!r} not in {sorted(data)[:8]}"
        data = data[split]
    out: Dict[str, List[str]] = {}
    for key, value in data.items():
        assert isinstance(value, (list, str)), f"{path}: unexpected value type for {key}"
        out[key] = value.split() if isinstance(value, str) else list(value)
    return out


def _to_ids(tokens: Sequence[str]) -> List[int]:
    """Phone ids under the 39 ARPAbet + SIL inventory (``emc.prior``'s one alias table)."""
    return _prior._to_ids(tokens)


def build_row(name: str, path: str, *, utt_ids: Sequence[str], split: Optional[str] = None,
              raw_path: Optional[str] = None, require_full_inventory: bool = False,
              secondary_na_reason: Optional[str] = None) -> PhoneRow:
    """One row RESTRICTED to the reference utterance set; SIL stripped from the primary strings.

    The restricted id set is asserted to equal the reference set exactly (no reference utterance
    may be missing); ids outside it are dropped and counted.  ``require_full_inventory`` asserts
    that the row realises all 39 ARPAbet monophones after the alias mapping -- the inventory check
    for a row whose symbols come from a foreign dictionary (the fairseq generator's).
    """
    raw_map = load_phone_json(path, split)
    missing = [u for u in utt_ids if u not in raw_map]
    assert not missing, f"{name}: {len(missing)} reference utterances missing, e.g. {missing[:3]}"
    ids_dropped = len(raw_map) - len(utt_ids)
    primary, stripped = [], 0
    for utt in utt_ids:
        ids = _to_ids(raw_map[utt])
        kept = [i for i in ids if i != _prior.SIL_ID]
        stripped += len(ids) - len(kept)
        primary.append(kept)
    sil_inclusive = None
    if raw_path is not None:
        sil_map = load_phone_json(raw_path, split)
        missing = [u for u in utt_ids if u not in sil_map]
        assert not missing, f"{name}: SIL-inclusive strings miss {len(missing)} utterances"
        sil_inclusive = [_to_ids(sil_map[utt]) for utt in utt_ids]
    row = PhoneRow(name, utt_ids, primary, sil_inclusive, stripped, ids_dropped=ids_dropped,
                   secondary_na_reason=secondary_na_reason)
    outside = set(row.inventory()) - set(_prior.ARPABET_39)
    assert not outside, f"{name}: symbols outside the 39 ARPAbet monophones: {sorted(outside)}"
    if require_full_inventory:
        absent = set(_prior.ARPABET_39) - set(row.inventory())
        assert not absent, f"{name}: the mapped inventory misses {sorted(absent)} of the 39 ARPAbet phones"
    return row


# ---------------------------------------------------------------------------------------------
# text side
# ---------------------------------------------------------------------------------------------


def read_text_side(corpus_path: str, *, n_count_lines: int, n_held_lines: int, held_stride: int,
                   orders: Sequence[int], chunk: int = 50_000,
                   require_full_inventory: bool = True) -> Tuple[object, Dict[int, "np.ndarray"], dict]:
    """Fit the SIL-free trigram and count the SIL-free n-grams on the training prior's own window."""
    counts = _prior.NgramCounts.zeros()
    ngrams = {n: np.zeros(BASE ** n, dtype=np.int64) for n in orders}
    seen = np.zeros(_prior.N_TYPES, dtype=np.int64)
    buf: List[List[int]] = []
    n_lines = n_tokens = n_sil = 0

    def flush():
        nonlocal buf
        if not buf:
            return
        counts.add_lines(buf)
        for n in orders:
            ngrams[n] += ngram_counts(buf, n)
        buf = []

    limit = n_count_lines + n_held_lines
    for i, tokens in enumerate(_prior.read_phone_lines(corpus_path, limit=limit)):
        if not tokens:
            continue
        if held_stride and i % held_stride == 0:  # the lines PhoneNgramPriorJob holds out
            continue
        ids = [j for j in _to_ids(tokens) if j != _prior.SIL_ID]
        n_sil += len(tokens) - len(ids)
        if not ids:
            continue
        seen[np.asarray(ids, dtype=np.int64)] = 1
        n_lines += 1
        n_tokens += len(ids)
        buf.append(ids)
        if len(buf) >= chunk:
            flush()
    flush()
    inventory = [_prior.PHONES[i] for i in np.flatnonzero(seen).tolist()]
    assert set(inventory) <= set(_prior.ARPABET_39), "the SIL-stripped text side is not ARPAbet-39"
    assert not require_full_inventory or set(inventory) == set(_prior.ARPABET_39), (
        f"the SIL-stripped text side realises {len(inventory)} of the 39 ARPAbet monophones")
    fit = _prior.PhoneNgramPrior.from_counts(counts, meta={
        "corpus": str(corpus_path), "sil": "stripped", "smoothing": "interpolated Witten-Bell",
        "lines_counted": n_lines, "tokens_counted": n_tokens})
    stats = {"corpus": str(corpus_path), "lines_read": limit, "held_stride": held_stride,
             "lines_counted": n_lines, "tokens_counted": n_tokens, "sil_tokens_stripped": n_sil,
             "inventory": inventory, "smoothing": "interpolated Witten-Bell (trigram->bigram->unigram->uniform)",
             "ngram_types_observed": {str(n): int((ngrams[n] > 0).sum()) for n in orders}}
    return fit, ngrams, stats


# ---------------------------------------------------------------------------------------------
# per-row statistics + utterance-block bootstrap
# ---------------------------------------------------------------------------------------------


def trigram_log_probs(fit, ids: Sequence[int]) -> "np.ndarray":
    """``[len(ids)]`` trigram log-probabilities: two BOS context symbols, no EOS term.

    Indexes ``fit.log_tri`` directly instead of calling ``PhoneNgramPrior.per_token_log_probs``:
    that helper builds its ``h2`` context as ``[BOS, BOS] + ids[:-2]``, which for a ONE-token
    sequence has length 2 and broadcasts, returning (and so double counting) that token twice.
    Every other length agrees with it exactly (asserted in the fixture test).
    """
    arr = np.asarray(list(ids), dtype=np.int64)
    if arr.size == 0:
        return np.zeros(0, dtype=np.float64)
    h1 = np.concatenate([[_prior.BOS_ID], arr[:-1]])
    h2 = np.concatenate([[_prior.BOS_ID, _prior.BOS_ID], arr[:-2]])[: arr.size]
    return np.asarray(fit.log_tri)[h2 * _prior.N_CTX + h1, arr]


def _utt_logprobs(fit, sequences: Sequence["np.ndarray"]) -> Tuple["np.ndarray", "np.ndarray"]:
    """Per-utterance (sum of trigram log-probs, token count); BOS-padded context, no EOS term."""
    sums = np.zeros(len(sequences), dtype=np.float64)
    toks = np.zeros(len(sequences), dtype=np.float64)
    for i, seq in enumerate(sequences):
        sums[i] = float(trigram_log_probs(fit, seq).sum())
        toks[i] = float(seq.size)
    return sums, toks


def _row_ngram_table(sequences: Sequence["np.ndarray"], order: int, text_counts: "np.ndarray"):
    """Support of the row's n-grams plus the (utterance, type) incidence used by the bootstrap."""
    utt_idx, packed = [], []
    for i, seq in enumerate(sequences):
        p = pack_ngrams(seq, order)
        if p.size:
            packed.append(p)
            utt_idx.append(np.full(p.size, i, dtype=np.int64))
    if not packed:
        return None
    packed = np.concatenate(packed)
    utt_idx = np.concatenate(utt_idx)
    support = np.unique(packed)
    comp = np.searchsorted(support, packed)
    key = utt_idx * support.size + comp
    ukey, mult = np.unique(key, return_counts=True)
    total = float(text_counts.sum())
    q_support = text_counts[support].astype(np.float64) / max(total, 1.0)
    return {
        "support_size": int(support.size),
        "u_utt": (ukey // support.size).astype(np.int64),
        "u_type": (ukey % support.size).astype(np.int64),
        "mult": mult.astype(np.float64),
        "q_support": q_support,
        "q_outside": float(1.0 - q_support.sum()),
        "full_counts": np.bincount(comp, minlength=support.size).astype(np.float64),
    }


def _jsd_from_table(table, weights: Optional["np.ndarray"]) -> float:
    if weights is None:
        counts = table["full_counts"]
    else:
        counts = np.bincount(table["u_type"], weights=weights[table["u_utt"]] * table["mult"],
                             minlength=table["support_size"])
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return jensen_shannon_bits(counts / total, table["q_support"], table["q_outside"])


def _ci(samples: Sequence[float]) -> List[float]:
    arr = np.asarray([s for s in samples if np.isfinite(s)], dtype=np.float64)
    if arr.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]


def _stat(value: float, samples: Sequence[float]) -> dict:
    """Point estimate plus BOTH bootstrap interval conventions and the resampling bias.

    A resample covers only ~63% of the distinct utterances, which shrinks the observed n-gram
    support and therefore inflates a plug-in JSD: the percentile interval of a JSD (and of a JSD
    difference) sits systematically ABOVE the full-sample value.  ``bootstrap_bias`` (mean of the
    replicates minus the point estimate) measures that shift and ``ci95_basic`` is the
    reverse-percentile interval 2*value - [q97.5, q2.5], which recentres on the point estimate.
    Both are reported; neither is picked for the reader.
    """
    arr = np.asarray([s for s in samples if np.isfinite(s)], dtype=np.float64)
    lo, hi = _ci(samples)
    return {"value": float(value), "ci95": [lo, hi],
            "ci95_basic": [2.0 * float(value) - hi, 2.0 * float(value) - lo],
            "bootstrap_bias": (float(arr.mean()) - float(value)) if arr.size else float("nan"),
            "n_resamples": int(arr.size)}


def _matched_weights(lengths: "np.ndarray", order_idx: "np.ndarray", budget: float,
                     n_utts: int) -> "np.ndarray":
    """Utterance weights of the COUNT-MATCHED subsample (see "Count matching" in the docstring).

    ``order_idx`` is the shuffled utterance order (with repeats inside a bootstrap replicate);
    whole utterances are taken from its front and the prefix stops at the first utterance that
    would push the phone count past ``budget``.  A prefix rule, not a greedy fill: no utterance
    length is preferred, so the subsample is an unbiased shortening of the row.
    """
    cum = np.cumsum(lengths[order_idx])
    k = int(np.searchsorted(cum, budget, side="right"))
    return np.bincount(order_idx[:k], minlength=n_utts).astype(np.float64)


def _verdict(stat: dict, key: str) -> dict:
    """PASS/FAIL of one pre-registered comparison, from the banked numbers only."""
    spec = PREREGISTERED_COMPARISONS[key]
    out = {"tag": spec["tag"], "margin": spec["margin"], "unit": spec["unit"],
           "rule": spec["rule"], "direction": spec["direction"],
           "requires_ci_excludes_zero": spec["ci_excludes_zero"]}
    value = stat.get("value", float("nan"))
    if not stat.get("available", True) or not np.isfinite(value):
        out.update({"point_meets_margin": None, "ci_excludes_zero": None, "verdict": "n/a"})
        return out
    lo, hi = stat["ci95"]
    meets = bool(value >= spec["margin"]) if spec["rule"] == "ge" else bool(abs(value) <= spec["margin"])
    excludes = bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0.0 or hi < 0.0))
    ok = meets and (excludes or not spec["ci_excludes_zero"])
    out.update({"point_meets_margin": meets, "ci_excludes_zero": excludes,
                "verdict": "PASS" if ok else "FAIL"})
    return out


def analyse(rows: Sequence[PhoneRow], *, fit, text_ngrams: Dict[int, "np.ndarray"],
            training_prior=None, orders: Sequence[int] = (1, 2, 3, 4),
            n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Every number of the read, count matched and with paired utterance-block bootstrap CIs.

    Primary statistics are computed on the count-matched subsample of each row (module docstring,
    "Count matching"); the unmatched full-count statistics are computed as well and banked as the
    secondary read.
    """
    n_utts = rows[0].n_utts
    for row in rows:
        assert row.n_utts == n_utts, f"{row.name}: {row.n_utts} utterances, expected {n_utts}"

    per_row = {}
    for row in rows:
        entry = {"n_utts": row.n_utts, "n_phones": row.n_phones, "sil_stripped": row.sil_stripped,
                 "ids_dropped": row.ids_dropped, "inventory": row.inventory(),
                 "secondary_na_reason": row.secondary_na_reason}
        sums, toks = _utt_logprobs(fit, row.primary)
        entry["_lp"] = (sums, toks)
        entry["_len"] = np.asarray([float(s.size) for s in row.primary], dtype=np.float64)
        entry["_tables"] = {n: _row_ngram_table(row.primary, n, text_ngrams[n]) for n in orders}
        if row.raw is not None and training_prior is not None:
            entry["_lp_raw"] = _utt_logprobs(training_prior, row.raw)
            entry["n_tokens_sil_inclusive"] = int(entry["_lp_raw"][1].sum())
        per_row[row.name] = entry

    # the pre-registered common phone budget: the smallest row decides how long every row may be
    budget = float(min(row.n_phones for row in rows))

    def weighted_lp(entry, key, w):
        sums, toks = entry[key]
        return float((w * sums).sum() / max(float((w * toks).sum()), 1.0))

    ones = np.ones(n_utts, dtype=np.float64)
    rng = np.random.default_rng(seed)                 # the utterance-block resamples
    rng_match = np.random.default_rng([int(seed), 1])  # the count-matching shuffles, same seed
    order0 = rng_match.permutation(n_utts)
    w_match0 = {name: _matched_weights(entry["_len"], order0, budget, n_utts)
                for name, entry in per_row.items()}

    boot = {name: {"logprob": [], "logprob_matched": [], "logprob_raw": [],
                   "jsd": {n: [] for n in orders}, "jsd_matched": {n: [] for n in orders}}
            for name in per_row}
    budgets = []
    for _ in range(int(n_bootstrap)):
        draw = rng.integers(0, n_utts, size=n_utts)
        w = np.bincount(draw, minlength=n_utts).astype(np.float64)  # shared by every row: paired
        order_r = rng_match.permutation(draw)          # one matching order, shared by every row
        budget_r = min([budget] + [float(entry["_len"][draw].sum()) for entry in per_row.values()])
        budgets.append(budget_r)
        for name, entry in per_row.items():
            wm = _matched_weights(entry["_len"], order_r, budget_r, n_utts)
            boot[name]["logprob"].append(weighted_lp(entry, "_lp", w))
            boot[name]["logprob_matched"].append(weighted_lp(entry, "_lp", wm))
            if "_lp_raw" in entry:
                boot[name]["logprob_raw"].append(weighted_lp(entry, "_lp_raw", w))
            for n in orders:
                table = entry["_tables"][n]
                boot[name]["jsd"][n].append(
                    float("nan") if table is None else _jsd_from_table(table, w))
                boot[name]["jsd_matched"][n].append(
                    float("nan") if table is None else _jsd_from_table(table, wm))

    out_rows = {}
    for name, entry in per_row.items():
        rec = {k: v for k, v in entry.items() if not k.startswith("_")}
        wm0 = w_match0[name]
        rec["n_phones_matched"] = int((wm0 * entry["_len"]).sum())
        rec["n_utts_matched"] = int(wm0.sum())
        rec["mean_logprob_per_phone_matched"] = _stat(weighted_lp(entry, "_lp", wm0),
                                                      boot[name]["logprob_matched"])
        rec["mean_logprob_per_phone"] = _stat(weighted_lp(entry, "_lp", ones),
                                              boot[name]["logprob"])
        rec["jsd_matched"], rec["jsd"] = {}, {}
        for n in orders:
            table = entry["_tables"][n]
            rec["jsd_matched"][str(n)] = _stat(
                float("nan") if table is None else _jsd_from_table(table, wm0),
                boot[name]["jsd_matched"][n])
            rec["jsd"][str(n)] = _stat(
                float("nan") if table is None else _jsd_from_table(table, None),
                boot[name]["jsd"][n])
        rec["secondary_mean_logprob_per_token_sil_inclusive"] = (
            _stat(weighted_lp(entry, "_lp_raw", ones), boot[name]["logprob_raw"])
            if "_lp_raw" in entry else None)
        out_rows[name] = rec

    matching = {
        "budget_phones": budget,
        "budget_row": min(out_rows, key=lambda nm: out_rows[nm]["n_phones"]),
        "rule": ("whole utterances taken from one shuffled order shared by the rows, prefix "
                 "truncated at the first utterance that would exceed the budget"),
        "applies_to": "every primary JSD, every primary mean log-prob and all four comparisons",
        "seed_stream": [int(seed), 1],
        "replicate_budget_mean": float(np.mean(budgets)) if budgets else float("nan"),
        "replicate_budget_min": float(np.min(budgets)) if budgets else float("nan"),
        "n_phones_full": {nm: out_rows[nm]["n_phones"] for nm in out_rows},
        "n_phones_matched": {nm: out_rows[nm]["n_phones_matched"] for nm in out_rows},
        "n_utts_matched": {nm: out_rows[nm]["n_utts_matched"] for nm in out_rows},
    }
    return {"rows": out_rows, "_boot": boot, "count_matching": matching, "_point": {
        name: {"logprob": out_rows[name]["mean_logprob_per_phone"]["value"],
               "logprob_matched": out_rows[name]["mean_logprob_per_phone_matched"]["value"],
               "jsd": {n: out_rows[name]["jsd"][str(n)]["value"] for n in orders},
               "jsd_matched": {n: out_rows[name]["jsd_matched"][str(n)]["value"] for n in orders}}
        for name in out_rows}}


def _difference(result: dict, left: str, right: str, kind: str, order: Optional[int] = None,
                *, matched: bool = True) -> dict:
    """Paired difference left - right with a bootstrap CI, or an explicit n/a naming what is absent.

    ``matched=True`` (the primary read) differences the COUNT-MATCHED statistics: inside one
    replicate both rows are resampled with the same utterance weights and then subsampled to the
    same phone budget from the same shuffled order, so the difference is paired twice over.
    ``matched=False`` gives the unmatched full-count difference, banked as the secondary.
    """
    absent = [name for name in (left, right) if name not in result["rows"]]
    if absent:
        return {"available": False, "missing": absent, "left": left, "right": right, "kind": kind,
                "order": order, "count_matched": matched}
    key = ("logprob" if kind == "logprob" else "jsd") + ("_matched" if matched else "")
    if kind == "logprob":
        point = result["_point"][left][key] - result["_point"][right][key]
        samples = [a - b for a, b in zip(result["_boot"][left][key], result["_boot"][right][key])]
    else:
        point = result["_point"][left][key][order] - result["_point"][right][key][order]
        samples = [a - b for a, b in zip(result["_boot"][left][key][order],
                                         result["_boot"][right][key][order])]
    return {"available": True, "left": left, "right": right, "kind": kind, "order": order,
            "count_matched": matched, **_stat(point, samples)}


def _fmt(stat: Optional[dict], digits: int = 4, interval: str = "ci95",
         na: Optional[str] = None) -> str:
    """``value [lo, hi]`` under the requested interval convention (see ``_stat``)."""
    if stat is None:
        return na or "n/a"
    if not stat.get("available", True):
        return f"n/a (missing row: {', '.join(stat['missing'])})"
    lo, hi = stat[interval]
    return f"{stat['value']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def render_summary(result: dict, *, orders: Sequence[int], row_order: Sequence[str],
                   spec: dict) -> str:
    rows = result["rows"]
    match = result["count_matching"]
    comp = result["comparisons"]
    lines = ["# SAE 4A step 1 -- n-gram mode-seeking check", "",
             "Training-free read; conventions and pre-registered comparisons are in the docstring of",
             "`recipe/i6_experiments/users/wu/experiments/unsupervised_asr/ngram_mode_seeking.py`.", "",
             f"- text side: {spec['text_side']['lines_counted']} lines / "
             f"{spec['text_side']['tokens_counted']} phones, SIL stripped "
             f"({spec['text_side']['sil_tokens_stripped']} SIL tokens removed)",
             f"- estimator: {spec['text_side']['smoothing']}, natural log; JSD base 2 (bits)",
             f"- bootstrap: {spec['n_bootstrap']} utterance-block resamples, seed {spec['seed']}, "
             "one shared resample per replicate so every difference is paired",
             "- interval conventions (fixed before the read): a per-row mean log-prob carries the "
             "percentile CI95; a per-row JSD carries the plug-in point estimate with the "
             "REVERSE-PERCENTILE (bias-corrected) CI95, because resampling utterances shrinks the "
             "observed n-gram support and inflates a plug-in JSD; every decisive comparison is a "
             "per-resample difference with the percentile CI95. The json carries both intervals "
             "and the bootstrap bias for every statistic.",
             "- COUNT MATCHING (primary, fixed before the read): a plug-in JSD grows as a row's "
             "token count shrinks and the rows differ in length, so every primary statistic and "
             f"all four comparisons are computed on a subsample of whole utterances down to a "
             f"common budget of {int(match['budget_phones'])} phones -- the smallest row "
             f"({match['budget_row']}) -- drawn from one shuffled utterance order shared by the "
             f"rows, for the point estimate and inside every bootstrap replicate (replicate "
             f"budget: mean {match['replicate_budget_mean']:.0f}, min "
             f"{match['replicate_budget_min']:.0f} phones). The unmatched full-count numbers are "
             "the secondary table below and are NOT the pre-registered read.", ""]
    header = ("| row | utts | phones (full) | phones (matched) | mean log P3 per phone "
              "(nats, matched) | " + " | ".join(f"JSD n={n} (matched)" for n in orders) +
              " | secondary log P3 per token (SIL-incl.) |")
    lines += [header, "|" + "---|" * (5 + len(orders) + 1)]
    for name in row_order:
        if name not in rows:
            cells = [name, "n/a", "n/a", "n/a", "n/a (row absent)"]
            cells += ["n/a (row absent)" for _ in orders] + ["n/a"]
            lines.append("| " + " | ".join(cells) + " |")
            continue
        rec = rows[name]
        cells = [name, str(rec["n_utts"]), str(rec["n_phones"]), str(rec["n_phones_matched"]),
                 _fmt(rec["mean_logprob_per_phone_matched"])]
        cells += [_fmt(rec["jsd_matched"][str(n)], interval="ci95_basic") for n in orders]
        cells.append(_fmt(rec["secondary_mean_logprob_per_token_sil_inclusive"],
                          na=f"n/a: {rec['secondary_na_reason']}" if rec.get("secondary_na_reason")
                          else "n/a"))
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Pre-registered comparisons (count matched; paired-difference bootstrap, "
                  "percentile CI95)", "",
              "| # | comparison | value [CI95] | pre-registered | verdict | phones full -> matched |",
              "|---|---|---|---|---|---|"]
    tags = (("a", "ep4_minus_gan_logprob", "mean SIL-free trigram log-prob per phone (nats)"),
            ("b", "ep4_minus_gold_jsd4", "4-gram JSD (bits)"),
            ("c", "ep4_minus_gan_jsd4", "4-gram JSD (bits)"),
            ("d", "gan_minus_gold_jsd4", "4-gram JSD (bits)"))
    for tag, key, what in tags:
        stat = comp[key]
        pre = stat.get("prereg", {})
        counts = "; ".join(f"{nm} {rows[nm]['n_phones']} -> {rows[nm]['n_phones_matched']}"
                           for nm in (stat["left"], stat["right"]) if nm in rows) or "n/a"
        lines.append(f"| ({tag}) | {stat['left']} minus {stat['right']}, {what} | {_fmt(stat)} | "
                     f"{pre.get('direction', 'n/a')} | {pre.get('verdict', 'n/a')} | {counts} |")
    lines += ["", "Each comparison above is the difference of the two rows inside the SAME "
                  "resample and the SAME count-matched budget, summarised by the percentile CI95 "
                  "of those differences. PASS = the point estimate meets the pre-registered "
                  "margin, and for (b) and (c) additionally the difference CI95 excludes zero "
                  "(SAE_4A_attrib.md, design-review amendments 2026-09-19)."]

    lines += ["", "## Secondary: unmatched full-count statistics (not the pre-registered read)", "",
              "| row | phones | mean log P3 per phone (nats) | " +
              " | ".join(f"JSD n={n}" for n in orders) + " |",
              "|" + "---|" * (3 + len(orders))]
    for name in row_order:
        if name not in rows:
            lines.append("| " + " | ".join([name, "n/a", "n/a"] + ["n/a" for _ in orders]) + " |")
            continue
        rec = rows[name]
        cells = [name, str(rec["n_phones"]), _fmt(rec["mean_logprob_per_phone"])]
        cells += [_fmt(rec["jsd"][str(n)], interval="ci95_basic") for n in orders]
        lines.append("| " + " | ".join(cells) + " |")
    unmatched = "; ".join(f"({tag}) {_fmt(comp[key + '_unmatched'])}" for tag, key, _ in tags)
    lines += ["", f"Same four comparisons WITHOUT count matching: {unmatched}."]

    ref = comp["jsd4_reference_line"]
    per_row = ", ".join(f"{name} {value:.4f}" for name, value in ref["rows_jsd4"].items())
    lines += ["", f"Reference line (descriptive only, decides nothing): Lin's gold-vs-text 4-gram JSD "
                  f"{ref['value']} is corpus-size dependent. Count-matched 4-gram JSD by row: {per_row}."]
    if spec.get("absent_rows"):
        lines += ["", f"Rows absent from this read: {', '.join(spec['absent_rows'])} "
                      "(their comparisons are n/a above)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# sisyphus job
# ---------------------------------------------------------------------------------------------


class NgramModeSeekingJob(Job):
    """CPU, in-process: fit the SIL-free trigram on the text side and score the banked decodes.

    :param rows: ordered ``{row name: cfg}`` with ``cfg`` keys ``path`` (the SIL-removed primary
        strings; ``None`` = the row is absent and its comparisons render as n/a), ``raw`` (the
        SIL-inclusive strings for the secondary read under ``prior_npz``), ``split`` (a split key
        inside a split-keyed json), ``require_full_inventory`` (assert the row realises all 39
        ARPAbet phones after the alias mapping; NOT set for the GAN row, whose generator never
        emits ZH -- code review 2026-09-19) and ``secondary_na`` (the reason rendered in the
        secondary column when the row has no SIL-inclusive strings).
    :param reference_row: the row whose utterance ids define the scored set; every other row is
        restricted to those ids and must cover all of them (paired bootstrap).
    :param corpus_phn: the phonemized text corpus behind the training trigram (SIL-inclusive; this
        job strips SIL).
    :param prior_npz: the TRAINED SIL-inclusive trigram, for the secondary read only.
    :param n_count_lines, n_held_lines, held_stride: the training prior's own corpus window.
    """

    def __init__(self, *, rows: Dict[str, dict], reference_row: str, corpus_phn: tk.Path,
                 prior_npz: tk.Path, comparison_rows: Dict[str, str],
                 n_count_lines: int = 1_000_000, n_held_lines: int = 10_000, held_stride: int = 101,
                 jsd_orders: Sequence[int] = (1, 2, 3, 4), n_bootstrap: int = 1000, seed: int = 0,
                 jsd4_reference_line: float = 0.27):
        self.rows = rows
        self.reference_row = reference_row
        self.corpus_phn = corpus_phn
        self.prior_npz = prior_npz
        self.comparison_rows = comparison_rows  # {"ep4": name, "gan": name, "gold": name}
        self.n_count_lines = int(n_count_lines)
        self.n_held_lines = int(n_held_lines)
        self.held_stride = int(held_stride)
        self.jsd_orders = tuple(int(n) for n in jsd_orders)
        self.n_bootstrap = int(n_bootstrap)
        self.seed = int(seed)
        self.jsd4_reference_line = float(jsd4_reference_line)
        assert reference_row in rows, f"reference row {reference_row!r} is not among {sorted(rows)}"
        self.out_json = self.output_path("ngram_mode_seeking.json")
        self.out_summary = self.output_path("summary.md")
        # measured on the real rows with a 200k-line text window: 0.33 GiB peak, seconds per
        # 100 resamples; the full 1.01M-line window scales the corpus pass linearly.
        self.rqmt = {"cpu": 2, "mem": 12, "time": 1}

    def tasks(self):
        yield Task("run", rqmt=self.rqmt)

    def run(self):
        spec_rows = {name: {k: (v.get_path() if isinstance(v, tk.Path) else v) for k, v in cfg.items()}
                     for name, cfg in self.rows.items()}
        ref_cfg = spec_rows[self.reference_row]
        ref_map = load_phone_json(ref_cfg["path"], ref_cfg.get("split"))
        utt_ids = sorted(ref_map)
        print(f"reference row {self.reference_row}: {len(utt_ids)} utterances", flush=True)

        rows, absent = [], []
        for name, cfg in spec_rows.items():
            if cfg.get("path") is None:
                absent.append(name)
                print(f"row {name}: ABSENT (no per-utterance hypotheses given)", flush=True)
                continue
            row = build_row(name, cfg["path"], utt_ids=utt_ids, split=cfg.get("split"),
                            raw_path=cfg.get("raw"),
                            require_full_inventory=bool(cfg.get("require_full_inventory")),
                            secondary_na_reason=cfg.get("secondary_na"))
            rows.append(row)
            print(f"row {name}: {row.n_utts} utts, {row.n_phones} phones, "
                  f"{len(row.inventory())} phone types, {row.sil_stripped} SIL stripped, "
                  f"{row.ids_dropped} ids outside the reference set dropped", flush=True)
        ref = next(r for r in rows if r.name == self.reference_row)
        missing_inv = set(_prior.ARPABET_39) - set(ref.inventory())
        assert not missing_inv, f"the reference row misses {sorted(missing_inv)} of the 39 ARPAbet phones"

        fit, text_ngrams, text_stats = read_text_side(
            self.corpus_phn.get_path(), n_count_lines=self.n_count_lines,
            n_held_lines=self.n_held_lines, held_stride=self.held_stride, orders=self.jsd_orders)
        print("text side:", json.dumps({k: v for k, v in text_stats.items() if k != "inventory"}),
              flush=True)
        training_prior = _prior.PhoneNgramPrior.load(self.prior_npz.get_path())

        result = analyse(rows, fit=fit, text_ngrams=text_ngrams, training_prior=training_prior,
                         orders=self.jsd_orders, n_bootstrap=self.n_bootstrap, seed=self.seed)

        ep4, gan, gold = (self.comparison_rows[k] for k in ("ep4", "gan", "gold"))
        # Primary = count matched (module docstring, "Count matching"); the unmatched full-count
        # difference of the same pair is banked beside it as the secondary.
        comparisons = {}
        for key, left, right, kind, order in (
                ("ep4_minus_gan_logprob", ep4, gan, "logprob", None),
                ("ep4_minus_gold_jsd4", ep4, gold, "jsd", 4),
                ("ep4_minus_gan_jsd4", ep4, gan, "jsd", 4),
                ("gan_minus_gold_jsd4", gan, gold, "jsd", 4)):
            stat = _difference(result, left, right, kind, order, matched=True)
            stat["prereg"] = _verdict(stat, key)
            comparisons[key] = stat
            comparisons[key + "_unmatched"] = _difference(result, left, right, kind, order,
                                                          matched=False)
        # Descriptive only (design review 2026-09-19): Lin's gold-vs-text value is corpus-size
        # dependent, so it is reported beside the 4-gram JSDs and decides nothing.
        comparisons["jsd4_reference_line"] = {
            "value": self.jsd4_reference_line, "decides": False,
            "note": "Lin's gold-vs-text 4-gram JSD; corpus-size dependent, descriptive only",
            "rows_jsd4": {name: rec["jsd_matched"]["4"]["value"] for name, rec in result["rows"].items()},
            "rows_jsd4_unmatched": {name: rec["jsd"]["4"]["value"] for name, rec in result["rows"].items()}}
        result["comparisons"] = comparisons
        print("count matching:", json.dumps(result["count_matching"]), flush=True)

        spec = {"rows": spec_rows, "reference_row": self.reference_row,
                "comparison_rows": self.comparison_rows, "absent_rows": absent,
                "corpus_phn": self.corpus_phn.get_path(), "prior_npz": self.prior_npz.get_path(),
                "n_count_lines": self.n_count_lines, "n_held_lines": self.n_held_lines,
                "held_stride": self.held_stride, "jsd_orders": list(self.jsd_orders),
                "n_bootstrap": self.n_bootstrap, "seed": self.seed,
                "jsd4_reference_line": self.jsd4_reference_line, "text_side": text_stats,
                "conventions": {
                    "primary": "SIL removed from every row and from the text side",
                    "logprob": "natural log, two BOS context symbols, no EOS term, token weighted",
                    "jsd": "base 2 (bits), n-grams inside one utterance/line, pooled over utterances",
                    "bootstrap": "utterance-block, shared resample across rows (paired), CI95 percentile",
                    "count_matching": ("primary JSDs, primary mean log-probs and all four "
                                       "comparisons are computed on a whole-utterance subsample of "
                                       "every row down to the smallest row's phone count, in the "
                                       "point estimate and in every replicate; the unmatched "
                                       "full-count statistics are banked as the secondary"),
                    "prereg": "PASS = point estimate meets the margin; (b) and (c) also need the "
                              "difference CI95 to exclude zero",
                    "secondary": "mean log P3 per token under the trained SIL-inclusive trigram"}}
        payload = {"spec": spec, "rows": result["rows"], "comparisons": comparisons,
                   "count_matching": result["count_matching"],
                   "prereg": PREREGISTERED_COMPARISONS}
        with open(self.out_json.get_path(), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        summary = render_summary(result, orders=self.jsd_orders, row_order=list(spec_rows),
                                 spec=spec)
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write(summary)
        print(summary, flush=True)
