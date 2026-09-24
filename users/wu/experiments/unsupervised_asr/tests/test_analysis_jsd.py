"""CPU tests of analysis.jsd (the D18 JS rows) on tiny synthetic inputs."""

from __future__ import annotations

import json
import os

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis import jsd as M
from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as _prior


def test_js_divergence_bounds_and_rows():
    p = np.array([3.0, 1.0, 0.0, 2.0])
    assert M.js_divergence(p, 2 * p) == pytest.approx(0.0, abs=1e-15)
    assert M.js_divergence([1.0, 0.0], [0.0, 1.0]) == pytest.approx(1.0)  # bits: disjoint = 1
    rng = np.random.RandomState(0)
    a = rng.randint(0, 5, size=(6, 7)).astype(float) + 0.5 * (rng.rand(6, 7) > 0.5)
    q = rng.rand(7)
    rows = M.js_rows(a, q)
    for i in range(len(a)):
        assert rows[i] == pytest.approx(M.js_divergence(a[i], q), abs=1e-14)


def test_bootstrap_weights_draw_like_cluster_bootstrap():
    from i6_experiments.users.wu.experiments.unsupervised_asr.analysis.paired import cluster_bootstrap

    clusters = [np.array(c) for c in ([0, 1], [2], [3, 4, 5], [6])]
    values = np.array([1.0, 2.0, 7.0, 0.5, 0.25, 4.0, 3.0])
    w = M.bootstrap_weights(len(clusters), n_boot=30, seed=5)
    assert w.shape == (30, 4) and (w.sum(1) == 4).all()
    # the same multiplicities give cluster_bootstrap's row-sum statistic
    sums = np.array([sum(values[c].sum() for c in clusters[i:i + 1]) for i in range(4)])
    means = (w @ sums) / (w @ np.array([len(c) for c in clusters]))
    np.testing.assert_allclose(means, cluster_bootstrap(values, clusters, n_boot=30, seed=5), rtol=1e-12)
    uni = M.speaker_counts({"a-1": ["AA", "AE"], "b-1": ["AE"]}, [np.array([0]), np.array([1])],
                           ["a-1", "b-1"], ["AA", "AE"])[0]
    np.testing.assert_array_equal(uni, [[1, 1], [0, 1]])


def test_js_rows_read_job_on_tiny_decodes(tmp_path):
    tmp = str(tmp_path)
    counts = _prior.NgramCounts.zeros()
    counts.add_lines([_prior._to_ids(l.split()) for l in ["AA AE AA", "SIL AE AA AE SIL", "AA AA AE"]])
    _prior.PhoneNgramPrior.from_counts(counts).save(os.path.join(tmp, "prior.npz"))
    gold = {"10-1-0": ["AA", "AE"], "10-1-1": ["AE", "AA", "AE"], "20-1-0": ["AA", "AA"]}
    json.dump({"tiny": gold}, open(os.path.join(tmp, "gold.json"), "w"))
    runs = {
        "base": {t: ["SIL"] + g[::-1][:2] for t, g in gold.items()},  # >= 1 bigram each
        "cand": {t: ["SIL"] + g + ["SIL"] for t, g in gold.items()},
    }
    decodes = {}
    for name, raw in runs.items():
        json.dump(raw, open(os.path.join(tmp, f"{name}_raw.json"), "w"))
        json.dump({t: [p for p in r if p != "SIL"] for t, r in raw.items()},
                  open(os.path.join(tmp, f"{name}_phones.json"), "w"))
        decodes[name] = {4: {"raw": tk.Path(os.path.join(tmp, f"{name}_raw.json")),
                             "phones": tk.Path(os.path.join(tmp, f"{name}_phones.json"))}}
    job = M.JsRowsReadJob(decodes=decodes,
                          rows=[{"tag": "cand-base", "cand": "cand", "cand_epoch": 4, "base": "base",
                                 "base_epoch": 4}],
                          gold=tk.Path(os.path.join(tmp, "gold.json")), split="tiny", expected_utterances=3,
                          prior_npz=tk.Path(os.path.join(tmp, "prior.npz")), n_boot=40, seed=0)
    job.out_json = tk.Path(os.path.join(tmp, "js_rows.json"))
    job.out_report = tk.Path(os.path.join(tmp, "report.txt"))
    job.run()
    rec = json.load(open(os.path.join(tmp, "js_rows.json")))
    cand = rec["decodes"]["cand"]["4"]["b3"]
    # the candidate decodes the gold exactly: zero JS to gold, and its SIL share is 6 of 13 tokens
    assert cand["js_gold_unigram"] == pytest.approx(0.0, abs=1e-15)
    assert cand["js_gold_bigram"] == pytest.approx(0.0, abs=1e-15)
    assert (cand["sil_tokens"], cand["raw_tokens"], cand["phone_tokens"]) == (6, 13, 7)
    row = rec["rows"][0]
    assert row["djs_gold_unigram"]["mean"] < 0 and row["djs_gold_bigram"]["mean"] < 0
    assert rec["references"] == {"n_speakers": 2, "n_utterances": 3}
    assert open(os.path.join(tmp, "report.txt")).read().startswith(M.JS_ROWS_CONVENTION)
