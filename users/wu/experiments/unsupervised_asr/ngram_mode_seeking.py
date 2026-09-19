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
                  row is asserted to be inside the 39 ARPAbet monophones; the text side and the
                  reference (gold) row are asserted to realise all 39.
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
Secondary         mean log P3 per token under the TRAINING SIL-INCLUSIVE trigram (``prior_npz``) on
                  the SIL-inclusive strings, for the model rows that have one; gold is n/a.
Uncertainty       utterance-block bootstrap, ``n_bootstrap`` resamples, ``seed``, CI95 as the
                  2.5/97.5 percentiles, on every mean log-prob and every JSD.  The SAME resampled
                  utterance multiset is used for every row inside a replicate, so the rendered
                  differences are paired.  The text side is fixed (not resampled): the CIs describe
                  utterance sampling of the rows only.
Rendered          ``summary.md`` MUST render, explicitly and each as a PAIRED-difference bootstrap
                  with its CI95 (SAE_4A_attrib.md "Design-review amendments", 2026-09-19):
                    (a) ep4 mean SIL-free trigram log-prob per phone MINUS the GAN's,
                    (b) ep4 4-gram JSD MINUS gold's,
                    (c) ep4 4-gram JSD MINUS the GAN's,
                    (d) GAN 4-gram JSD MINUS gold's.
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
                 raw: Optional[Sequence[Sequence[int]]] = None, sil_stripped: int = 0):
        self.name = name
        self.utt_ids = list(utt_ids)
        self.primary = [np.asarray(s, dtype=np.int64) for s in primary]
        self.raw = None if raw is None else [np.asarray(s, dtype=np.int64) for s in raw]
        self.sil_stripped = int(sil_stripped)
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
              raw_path: Optional[str] = None) -> PhoneRow:
    """One row on the reference utterance set; SIL stripped from the primary strings."""
    raw_map = load_phone_json(path, split)
    missing = [u for u in utt_ids if u not in raw_map]
    assert not missing, f"{name}: {len(missing)} reference utterances missing, e.g. {missing[:3]}"
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
    row = PhoneRow(name, utt_ids, primary, sil_inclusive, stripped)
    outside = set(row.inventory()) - set(_prior.ARPABET_39)
    assert not outside, f"{name}: symbols outside the 39 ARPAbet monophones: {sorted(outside)}"
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


def analyse(rows: Sequence[PhoneRow], *, fit, text_ngrams: Dict[int, "np.ndarray"],
            training_prior=None, orders: Sequence[int] = (1, 2, 3, 4),
            n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Every number of the read, with paired utterance-block bootstrap CIs (see module docstring)."""
    n_utts = rows[0].n_utts
    for row in rows:
        assert row.n_utts == n_utts, f"{row.name}: {row.n_utts} utterances, expected {n_utts}"

    per_row = {}
    for row in rows:
        entry = {"n_utts": row.n_utts, "n_phones": row.n_phones, "sil_stripped": row.sil_stripped,
                 "inventory": row.inventory()}
        sums, toks = _utt_logprobs(fit, row.primary)
        entry["_lp"] = (sums, toks)
        entry["_tables"] = {n: _row_ngram_table(row.primary, n, text_ngrams[n]) for n in orders}
        if row.raw is not None and training_prior is not None:
            entry["_lp_raw"] = _utt_logprobs(training_prior, row.raw)
            entry["n_tokens_sil_inclusive"] = int(entry["_lp_raw"][1].sum())
        per_row[row.name] = entry

    def point_lp(key, entry):
        sums, toks = entry[key]
        return float(sums.sum() / max(toks.sum(), 1.0))

    boot = {name: {"logprob": [], "logprob_raw": [], "jsd": {n: [] for n in orders}}
            for name in per_row}
    rng = np.random.default_rng(seed)
    for _ in range(int(n_bootstrap)):
        draw = rng.integers(0, n_utts, size=n_utts)
        w = np.bincount(draw, minlength=n_utts).astype(np.float64)  # shared by every row: paired
        for name, entry in per_row.items():
            sums, toks = entry["_lp"]
            boot[name]["logprob"].append(float((w * sums).sum() / max((w * toks).sum(), 1.0)))
            if "_lp_raw" in entry:
                rsums, rtoks = entry["_lp_raw"]
                boot[name]["logprob_raw"].append(float((w * rsums).sum() / max((w * rtoks).sum(), 1.0)))
            for n in orders:
                table = entry["_tables"][n]
                boot[name]["jsd"][n].append(float("nan") if table is None else _jsd_from_table(table, w))

    out_rows = {}
    for name, entry in per_row.items():
        rec = {k: v for k, v in entry.items() if not k.startswith("_")}
        rec["mean_logprob_per_phone"] = _stat(point_lp("_lp", entry), boot[name]["logprob"])
        rec["jsd"] = {}
        for n in orders:
            table = entry["_tables"][n]
            rec["jsd"][str(n)] = _stat(float("nan") if table is None else _jsd_from_table(table, None),
                                       boot[name]["jsd"][n])
        rec["secondary_mean_logprob_per_token_sil_inclusive"] = (
            _stat(point_lp("_lp_raw", entry), boot[name]["logprob_raw"]) if "_lp_raw" in entry else None)
        out_rows[name] = rec
    return {"rows": out_rows, "_boot": boot, "_point": {
        name: {"logprob": out_rows[name]["mean_logprob_per_phone"]["value"],
               "jsd": {n: out_rows[name]["jsd"][str(n)]["value"] for n in orders}} for name in out_rows}}


def _difference(result: dict, left: str, right: str, kind: str, order: Optional[int] = None) -> dict:
    """Paired difference left - right with a bootstrap CI, or an explicit n/a naming what is absent."""
    absent = [name for name in (left, right) if name not in result["rows"]]
    if absent:
        return {"available": False, "missing": absent, "left": left, "right": right, "kind": kind,
                "order": order}
    if kind == "logprob":
        point = result["_point"][left]["logprob"] - result["_point"][right]["logprob"]
        samples = [a - b for a, b in zip(result["_boot"][left]["logprob"], result["_boot"][right]["logprob"])]
    else:
        point = result["_point"][left]["jsd"][order] - result["_point"][right]["jsd"][order]
        samples = [a - b for a, b in zip(result["_boot"][left]["jsd"][order],
                                         result["_boot"][right]["jsd"][order])]
    return {"available": True, "left": left, "right": right, "kind": kind, "order": order,
            **_stat(point, samples)}


def _fmt(stat: Optional[dict], digits: int = 4) -> str:
    if stat is None:
        return "n/a"
    if not stat.get("available", True):
        return f"n/a (missing row: {', '.join(stat['missing'])})"
    lo, hi = stat["ci95"]
    return f"{stat['value']:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def render_summary(result: dict, *, orders: Sequence[int], row_order: Sequence[str],
                   spec: dict) -> str:
    rows = result["rows"]
    lines = ["# SAE 4A step 1 -- n-gram mode-seeking check", "",
             "Training-free read; conventions and pre-registered comparisons are in the docstring of",
             "`recipe/i6_experiments/users/wu/experiments/unsupervised_asr/ngram_mode_seeking.py`.", "",
             f"- text side: {spec['text_side']['lines_counted']} lines / "
             f"{spec['text_side']['tokens_counted']} phones, SIL stripped "
             f"({spec['text_side']['sil_tokens_stripped']} SIL tokens removed)",
             f"- estimator: {spec['text_side']['smoothing']}, natural log; JSD base 2 (bits)",
             f"- bootstrap: {spec['n_bootstrap']} utterance resamples, seed {spec['seed']}, CI95 percentile, "
             "paired across rows", ""]
    header = "| row | utts | phones | mean log P3 per phone (nats) | " + \
             " | ".join(f"JSD n={n}" for n in orders) + " | secondary log P3 per token (SIL-incl.) |"
    lines += [header, "|" + "---|" * (4 + len(orders) + 1)]
    for name in row_order:
        if name not in rows:
            cells = [name, "n/a", "n/a", "n/a (row absent)"] + ["n/a (row absent)" for _ in orders]
            lines.append("| " + " | ".join(cells + ["n/a"]) + " |")
            continue
        rec = rows[name]
        cells = [name, str(rec["n_utts"]), str(rec["n_phones"]),
                 _fmt(rec["mean_logprob_per_phone"])]
        cells += [_fmt(rec["jsd"][str(n)]) for n in orders]
        cells.append(_fmt(rec["secondary_mean_logprob_per_token_sil_inclusive"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Pre-registered comparisons (paired-difference bootstrap, CI95)", ""]
    comp = result["comparisons"]
    for tag, key, what in (("a", "ep4_minus_gan_logprob", "mean SIL-free trigram log-prob per phone (nats)"),
                           ("b", "ep4_minus_gold_jsd4", "4-gram JSD (bits)"),
                           ("c", "ep4_minus_gan_jsd4", "4-gram JSD (bits)"),
                           ("d", "gan_minus_gold_jsd4", "4-gram JSD (bits)")):
        stat = comp[key]
        left, right = stat["left"], stat["right"]
        lines.append(f"({tag}) {left} minus {right}, {what}: {_fmt(stat)}")
        if stat.get("available"):
            blo, bhi = stat["ci95_basic"]
            lines.append(f"      bootstrap bias {stat['bootstrap_bias']:+.4f}; "
                         f"bias-corrected (reverse-percentile) CI95 [{blo:.4f}, {bhi:.4f}]")
    lines += ["", "CI95 above is the percentile interval of the paired difference replicates. "
                  "Resampling utterances shrinks the observed n-gram support, which inflates a "
                  "plug-in JSD, so a JSD percentile interval sits above the full-sample value; the "
                  "reverse-percentile interval recentres on it. Both are in the json for every "
                  "statistic."]
    ref = comp["jsd4_reference_line"]
    per_row = ", ".join(f"{name} {value:.4f}" for name, value in ref["rows_jsd4"].items())
    lines += ["", f"Reference line (descriptive only, decides nothing): Lin's gold-vs-text 4-gram JSD "
                  f"{ref['value']} is corpus-size dependent. 4-gram JSD by row: {per_row}."]
    if spec.get("absent_rows"):
        lines += ["", f"Rows absent from this read: {', '.join(spec['absent_rows'])} "
                      "(their comparisons are n/a above)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------
# sisyphus job
# ---------------------------------------------------------------------------------------------


class NgramModeSeekingJob(Job):
    """CPU, in-process: fit the SIL-free trigram on the text side and score the banked decodes.

    :param rows: ordered ``{row name: {"path": tk.Path, "split": str|None, "raw": tk.Path|None}}``;
        ``path`` holds the SIL-removed (primary) strings, ``raw`` the SIL-inclusive ones (secondary,
        scored under ``prior_npz``), ``split`` selects a split inside a split-keyed json.
    :param reference_row: the row whose utterance ids define the scored set; every other row must
        cover it exactly (paired bootstrap).
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
                            raw_path=cfg.get("raw"))
            rows.append(row)
            print(f"row {name}: {row.n_utts} utts, {row.n_phones} phones, "
                  f"{len(row.inventory())} phone types, {row.sil_stripped} SIL stripped", flush=True)
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
        comparisons = {
            "ep4_minus_gan_logprob": _difference(result, ep4, gan, "logprob"),
            "ep4_minus_gold_jsd4": _difference(result, ep4, gold, "jsd", 4),
            "ep4_minus_gan_jsd4": _difference(result, ep4, gan, "jsd", 4),
            "gan_minus_gold_jsd4": _difference(result, gan, gold, "jsd", 4),
            # Descriptive only (design review 2026-09-19): Lin's gold-vs-text value is corpus-size
            # dependent, so it is reported beside the 4-gram JSDs and decides nothing.
            "jsd4_reference_line": {
                "value": self.jsd4_reference_line, "decides": False,
                "note": "Lin's gold-vs-text 4-gram JSD; corpus-size dependent, descriptive only",
                "rows_jsd4": {name: rec["jsd"]["4"]["value"] for name, rec in result["rows"].items()}},
        }
        result["comparisons"] = comparisons

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
                    "secondary": "mean log P3 per token under the trained SIL-inclusive trigram"}}
        payload = {"spec": spec, "rows": result["rows"], "comparisons": comparisons}
        with open(self.out_json.get_path(), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        summary = render_summary(result, orders=self.jsd_orders, row_order=list(spec_rows),
                                 spec=spec)
        with open(self.out_summary.get_path(), "w") as fh:
            fh.write(summary)
        print(summary, flush=True)
