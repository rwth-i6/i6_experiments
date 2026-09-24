"""CPU tests of analysis.paired on tiny synthetic inputs."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis import paired as PP


def test_cluster_bootstrap_matches_explicit_resampling():
    values = np.random.RandomState(3).rand(12)
    clusters = [np.array(c) for c in ([0, 1, 2], [3, 4], [5, 6, 7, 8], [9], [10, 11])]
    boot = PP.cluster_bootstrap(values, clusters, n_boot=50, seed=7)
    rng = np.random.default_rng(7)
    expected = []
    for _ in range(50):
        picked = rng.integers(0, len(clusters), size=len(clusters))
        expected.append(values[np.concatenate([clusters[i] for i in picked])].mean())
    assert boot.dtype == np.float64 and boot.shape == (50,)
    np.testing.assert_array_equal(boot, np.array(expected))


def _paired(tmp, hyp_a, hyp_b, gold, split="tiny"):
    for fn, obj in (("a.json", hyp_a), ("b.json", hyp_b), ("gold.json", {split: gold})):
        json.dump(obj, open(os.path.join(tmp, fn), "w"))
    job = PP.PairedPerDeltaJob(per_a=tk.Path(os.path.join(tmp, "a.json")),
                               per_b=tk.Path(os.path.join(tmp, "b.json")),
                               gold=tk.Path(os.path.join(tmp, "gold.json")), split=split, name="B",
                               n_boot=100, seed=0)
    job.out_paired_per = tk.Path(os.path.join(tmp, "paired_per.json"))
    job.out_summary = tk.Path(os.path.join(tmp, "summary.txt"))
    job.run()
    return json.load(open(os.path.join(tmp, "paired_per.json")))


def test_paired_per_delta_on_tiny_split(tmp_path):
    gold = {f"{spk}-1-{k}": ["AA", "B", "K"] for spk in (10, 20, 30) for k in range(3)}
    hyp_a = {t: ["AA"] for t in gold}  # 2 deletions each
    hyp_b = {t: (["AA", "B", "K"] if t.startswith("10-") else ["AA"]) for t in gold}
    rec = _paired(str(tmp_path), hyp_a, hyp_b, gold)
    assert rec["utterances"] == 9 and rec["speakers"] == 3 and rec["ref_phones"] == 27
    assert rec["per_a"] == pytest.approx(18 / 27) and rec["per_b"] == pytest.approx(12 / 27)
    assert rec["delta_per"] == pytest.approx(-6 / 27)
    assert (rec["n_improved"], rec["n_worse"], rec["n_tied"]) == (3, 0, 6)
    lo, hi = rec["delta_per_ci95"]
    assert lo <= rec["delta_per"] <= hi <= 0.0
    assert open(os.path.join(str(tmp_path), "summary.txt")).read().startswith("tiny B vs init: delta_per = ")


def test_paired_per_refuses_partial_overlap(tmp_path):
    gold = {"10-1-0": ["AA"], "20-1-0": ["AA"]}
    with pytest.raises(AssertionError):
        _paired(str(tmp_path), {"10-1-0": ["AA"]}, {"10-1-0": ["AA"], "20-1-0": []}, gold)


def test_paired_per_asserts_registered_split_size(tmp_path):
    gold = {"10-1-0": ["AA"]}
    with pytest.raises(AssertionError, match="expected 2703"):
        _paired(str(tmp_path), {"10-1-0": ["AA"]}, {"10-1-0": ["AA"]}, gold, split="dev-clean")


# ===================================================================================================
# Priority-2 test (test plan 2026-09-24, T2.8): the paired CI re-implemented independently.
# ===================================================================================================


def _lev(a, b) -> int:
    """Unit-cost edit distance, independent of analysis.per (full table)."""
    d = [[i + j if i * j == 0 else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[-1][-1]


def _independent(hyp_a, hyp_b, gold, n_boot, seed):
    """The registered construction (PAIRED_PER_CONVENTION), written out: sorted tags, speakers =
    tag prefix before the first '-', clusters in sorted speaker order, speakers resampled with
    replacement by numpy's default_rng(seed), ratio of sums per resample, [2.5, 97.5] percentiles."""
    tags = sorted(gold)
    d_a = np.array([_lev(gold[t], hyp_a[t]) for t in tags], dtype=np.float64)
    d_b = np.array([_lev(gold[t], hyp_b[t]) for t in tags], dtype=np.float64)
    n = np.array([len(gold[t]) for t in tags], dtype=np.float64)
    speakers = sorted({t.split("-")[0] for t in tags})
    clusters = [np.array([k for k, t in enumerate(tags) if t.split("-")[0] == s], dtype=np.int64) for s in speakers]
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([clusters[i] for i in rng.integers(0, len(clusters), size=len(clusters))])
        boot[b] = (d_b[idx].sum() - d_a[idx].sum()) / n[idx].sum()
    macro = (d_b - d_a) / np.maximum(n, 1.0)
    return dict(ci=[float(x) for x in np.percentile(boot, [2.5, 97.5])],
                macro_ci=[float(x) for x in np.percentile(PP.cluster_bootstrap(macro, clusters, n_boot=n_boot,
                                                                               seed=seed), [2.5, 97.5])],
                delta=float((d_b.sum() - d_a.sum()) / n.sum()), macro=float(macro.mean()), speakers=len(clusters))


def _crafted(kind, rng):
    """Gold over 12 speakers x 1-5 utterances; B edits the gold (or A's hypothesis) per ``kind``."""
    phones = ["AA", "B", "K", "IY", "S", "T"]
    gold = {}
    for spk in range(12):
        for k in range(1 + spk % 5):
            gold[f"{100 + spk}-7-{k}"] = [phones[i] for i in rng.randint(0, 6, size=rng.randint(1, 9))]

    def noisy(seq, p):
        out = []
        for ph in seq:
            r = rng.rand()
            if r < p / 3:
                continue  # deletion
            out.append(phones[rng.randint(0, 6)] if r < 2 * p / 3 else ph)
            if r > 1 - p / 3:
                out.append(phones[rng.randint(0, 6)])  # insertion
        return out

    hyp_a = {t: noisy(g, 0.5) for t, g in gold.items()}
    if kind == "better":
        hyp_b = {t: noisy(g, 0.05) for t, g in gold.items()}
    elif kind == "worse":
        hyp_b = {t: noisy(g, 0.95) for t, g in gold.items()}
    else:  # "mixed": B = A with a few utterances nudged both ways
        hyp_b = dict(hyp_a)
        for k, t in enumerate(sorted(gold)[:6]):
            hyp_b[t] = list(gold[t]) if k % 2 == 0 else hyp_a[t] + ["AA", "B"]
    return gold, hyp_a, hyp_b


@pytest.mark.parametrize("kind, refines, excludes_zero", [
    ("better", True, True), ("worse", False, True), ("mixed", False, False)])
def test_t2_8_paired_ci_reimplemented(tmp_path, kind, refines, excludes_zero):
    rng = np.random.RandomState({"better": 1, "worse": 2, "mixed": 3}[kind])
    gold, hyp_a, hyp_b = _crafted(kind, rng)
    rec = _paired(str(tmp_path), hyp_a, hyp_b, gold)  # n_boot 100, seed 0
    ref = _independent(hyp_a, hyp_b, gold, n_boot=100, seed=0)
    print(f"T2.8 {kind}: delta {rec['delta_per']!r} CI {rec['delta_per_ci95']} macro CI {rec['delta_per_macro_ci95']}")
    assert rec["speakers"] == ref["speakers"] == 12
    # the job forms per_b - per_a (two divisions), the convention (sum d_B - sum d_A) / sum N: equal
    # up to one rounding
    assert rec["delta_per"] == pytest.approx(ref["delta"], abs=1e-15) and rec["delta_per_macro"] == ref["macro"]
    assert rec["delta_per_ci95"] == ref["ci"]  # bitwise
    assert rec["delta_per_macro_ci95"] == ref["macro_ci"]  # bitwise
    lo, hi = rec["delta_per_ci95"]
    assert rec["refines"] is refines and rec["refines"] == (hi < 0.0)
    assert rec["excludes_zero"] is excludes_zero and rec["excludes_zero"] == (lo > 0.0 or hi < 0.0)
