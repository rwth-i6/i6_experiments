"""Fixture tests for the SAE 4A step-1 n-gram mode-seeking reader (3 synthetic utterances).

Two properties, both checked against something that is NOT the production code path:
  * a row scored against ITS OWN n-gram distribution has JSD 0 at every order;
  * the mean SIL-free trigram log-prob per phone equals an explicit interpolated Witten-Bell
    computation written out here from literal counts (so a wrong BOS padding, a wrong backoff
    context, or a wrong token weighting fails the test).
"""

from __future__ import annotations

import json
import math
import os
import tempfile

from speech_llm.sae.emc import prior as _prior

from i6_experiments.users.wu.experiments.unsupervised_asr import ngram_mode_seeking as M

# The fixture: three text lines (SIL-inclusive, as the real corpus is) and three utterances.
TEXT_LINES = ["<SIL> AA AE AA <SIL>", "AA AE AA AE", "<SIL> AE AA AA"]
UTTS = {"u1": ["AA", "AE", "AA", "AE", "AA"], "u2": ["AE", "AA", "AE", "AA"],
        "u3": ["SIL", "AA", "AE", "AA", "AA"]}
UTT_IDS = ["u1", "u2", "u3"]


def _write(tmpdir: str, name: str, payload) -> str:
    path = os.path.join(tmpdir, name)
    with open(path, "w") as fh:
        if isinstance(payload, list):
            fh.write("\n".join(payload) + "\n")
        else:
            json.dump(payload, fh)
    return path


# -- the hand computation ----------------------------------------------------------------------
# SIL-stripped text lines: [AA AE AA], [AA AE AA AE], [AE AA AA].  Every count below is read off
# them by hand; V = 40 types, the uniform backoff is 1/V, and each line is padded with two BOS.
V = 40
UNI = {"AA": 6, "AE": 4}                                        # 10 tokens, 2 distinct
BI = {("BOS", "AA"): 2, ("BOS", "AE"): 1, ("AA", "AE"): 3, ("AE", "AA"): 3, ("AA", "AA"): 1}
TRI = {("BOS", "BOS", "AA"): 2, ("BOS", "BOS", "AE"): 1, ("BOS", "AA", "AE"): 2,
       ("AA", "AE", "AA"): 2, ("AE", "AA", "AE"): 1, ("BOS", "AE", "AA"): 1,
       ("AE", "AA", "AA"): 1}


def _p1(w: str) -> float:
    n, t = sum(UNI.values()), len(UNI)
    return (UNI.get(w, 0) + t * (1.0 / V)) / (n + t)


def _p2(w: str, h: str) -> float:
    rows = {k[1]: v for k, v in BI.items() if k[0] == h}
    n, t = sum(rows.values()), len(rows)
    if n == 0:
        return _p1(w)
    return (rows.get(w, 0) + t * _p1(w)) / (n + t)


def _p3(w: str, h2: str, h1: str) -> float:
    rows = {k[2]: v for k, v in TRI.items() if k[0] == h2 and k[1] == h1}
    n, t = sum(rows.values()), len(rows)
    if n == 0:
        return _p2(w, h1)
    return (rows.get(w, 0) + t * _p2(w, h1)) / (n + t)


def _hand_mean_logprob() -> float:
    total, tokens = 0.0, 0
    for utt in UTT_IDS:
        seq = [p for p in UTTS[utt] if p != "SIL"]          # the primary convention strips SIL
        ctx = ["BOS", "BOS"]
        for phone in seq:
            total += math.log(_p3(phone, ctx[-2], ctx[-1]))  # natural log, no EOS term
            ctx.append(phone)
        tokens += len(seq)
    return total / tokens                                    # token weighted, not per utterance


# -- tests -------------------------------------------------------------------------------------


def test_self_jsd_is_zero():
    """A row scored against its own pooled n-gram distribution has JSD 0 at n = 1..4."""
    seqs = [[_prior.PHONE2ID[p] for p in UTTS[u] if p != "SIL"] for u in UTT_IDS]
    for order in (1, 2, 3, 4):
        own = M.ngram_counts(seqs, order)
        table = M._row_ngram_table([__import__("numpy").asarray(s) for s in seqs], order, own)
        if table is None:                                    # no utterance is long enough
            continue
        assert abs(M._jsd_from_table(table, None)) < 1e-12, (order, M._jsd_from_table(table, None))


def test_mean_logprob_matches_hand_computation():
    with tempfile.TemporaryDirectory() as tmp:
        corpus = _write(tmp, "text.phn", TEXT_LINES)
        primary = _write(tmp, "hyp.json", {u: UTTS[u] for u in UTT_IDS})
        fit, text_ngrams, stats = M.read_text_side(
            corpus, n_count_lines=len(TEXT_LINES), n_held_lines=0, held_stride=0,
            orders=(1, 2, 3, 4), require_full_inventory=False)
        assert stats["lines_counted"] == 3 and stats["tokens_counted"] == 10
        assert stats["sil_tokens_stripped"] == 3
        row = M.build_row("fixture", primary, utt_ids=UTT_IDS)
        assert row.n_utts == 3 and row.n_phones == 13 and row.sil_stripped == 1
        result = M.analyse([row], fit=fit, text_ngrams=text_ngrams, orders=(1, 2, 3, 4),
                           n_bootstrap=16, seed=0)
        got = result["rows"]["fixture"]["mean_logprob_per_phone"]["value"]
        want = _hand_mean_logprob()
        assert abs(got - want) < 1e-12, (got, want)
        lo, hi = result["rows"]["fixture"]["mean_logprob_per_phone"]["ci95"]
        assert math.isfinite(lo) and math.isfinite(hi) and lo <= hi, (lo, hi)
        # every length but 1 must agree with the shared helper (see trigram_log_probs' docstring)
        for seq in ([0, 1, 0], [1, 0], [0, 1, 0, 1]):
            import numpy as np
            want_shared = fit.per_token_log_probs(np.asarray(seq), order=3)
            got_local = M.trigram_log_probs(fit, seq)
            if len(seq) > 1:
                assert abs(float(want_shared.sum()) - float(got_local.sum())) < 1e-12, seq
        single = M.trigram_log_probs(fit, [0])
        assert single.size == 1, single
        for order in ("1", "2", "3", "4"):
            jsd = result["rows"]["fixture"]["jsd"][order]["value"]
            assert 0.0 <= jsd <= 1.0 + 1e-12, (order, jsd)


def test_row_against_identical_text_side_has_zero_jsd():
    """End-to-end: when the text corpus IS the row, every reported JSD is 0."""
    lines = [" ".join(p for p in UTTS[u] if p != "SIL") for u in UTT_IDS]
    with tempfile.TemporaryDirectory() as tmp:
        corpus = _write(tmp, "text.phn", lines)
        primary = _write(tmp, "hyp.json", {u: [p for p in UTTS[u] if p != "SIL"] for u in UTT_IDS})
        fit, text_ngrams, _ = M.read_text_side(
            corpus, n_count_lines=len(lines), n_held_lines=0, held_stride=0, orders=(1, 2, 3, 4),
            require_full_inventory=False)
        row = M.build_row("fixture", primary, utt_ids=UTT_IDS)
        result = M.analyse([row], fit=fit, text_ngrams=text_ngrams, orders=(1, 2, 3, 4),
                           n_bootstrap=8, seed=0)
        for order in ("1", "2", "3", "4"):
            assert abs(result["rows"]["fixture"]["jsd"][order]["value"]) < 1e-12, order


if __name__ == "__main__":
    test_self_jsd_is_zero()
    test_mean_logprob_matches_hand_computation()
    test_row_against_identical_text_side_has_zero_jsd()
    print("ok")
