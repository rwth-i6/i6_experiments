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
