"""CPU tests of analysis.per and analysis.posterior on tiny synthetic inputs (no model forward)."""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis import per as P
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import PHONES


def _write_hdf(path, rows, dtype):
    """``{tag: array [T] or [T, D]}`` in RETURNN's HDF layout (inputs / seqLengths / seqTags)."""
    tags = list(rows)
    data = np.concatenate([np.asarray(rows[t], dtype=dtype) for t in tags], axis=0)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("inputs", data=data)
        fh.create_dataset("seqLengths", data=np.array([[len(rows[t]), 0] for t in tags], dtype="int32"))
        fh.create_dataset("seqTags", data=np.array([t.encode() for t in tags]))
    return tk.Path(path)


def _outputs(job, tmp, names):
    for attr, fn in names.items():
        setattr(job, attr, tk.Path(os.path.join(tmp, fn)))


def test_edit_counts():
    assert P.edit_counts([], []) == (0, 0, 0)
    assert P.edit_counts(["A"], []) == (0, 0, 1)
    assert P.edit_counts([], ["A", "B"]) == (0, 2, 0)
    assert P.edit_counts(["A", "C"], ["A", "B"]) == (1, 0, 0)
    assert P.edit_counts(["A", "B", "C"], ["A", "C"]) == (0, 0, 1)
    s, d, i = P.edit_counts(["X", "Y", "Z"], ["A", "B", "C", "D"])
    assert s + d + i == 4 and d >= 1


def test_blankfree_greedy_per_on_synthetic_posteriors(tmp_path):
    """2703 dev-clean utterances; utterance 0 decodes to its gold, every other one to SIL + AA."""
    tmp = str(tmp_path)
    sil, aa, ae = PHONES.index("SIL"), PHONES.index("AA"), PHONES.index("AE")
    tags = [f"{1000 + k // 10}-1-{k:04d}" for k in range(2703)]
    post, feats, orig, gold = {}, {}, {}, {}
    for k, tag in enumerate(tags):
        path = [sil, aa, aa, ae, sil] if k == 0 else [sil, sil, aa, aa]
        q = np.full((len(path), 40), -10.0, dtype=np.float32)
        q[np.arange(len(path)), path] = 0.0
        post[tag] = q
        feats[tag] = np.zeros((3 * len(path) - 1, 2), dtype=np.float16)  # (retained + 2) // 3 == T
        orig[tag] = np.array([4 * len(path)])
        gold[tag] = ["AA", "AE"] if k == 0 else ["AE"]
    json.dump({"dev-clean": gold}, open(os.path.join(tmp, "gold.json"), "w"))
    job = P.BlankfreeGreedyPerJob(
        posteriors=_write_hdf(os.path.join(tmp, "post.hdf"), post, "float32"),
        features=[_write_hdf(os.path.join(tmp, "feats.hdf"), feats, "float16")],
        originals=[_write_hdf(os.path.join(tmp, "orig.hdf"), orig, "int32")],
        gold=tk.Path(os.path.join(tmp, "gold.json")), split="dev-clean")
    _outputs(job, tmp, {"out_per": "per.json", "out_report": "per.txt", "out_hyps": "hyps.json",
                        "out_raw_hyps": "raw.json", "out_stats": "stats.json"})
    job.run()
    rec = json.load(open(os.path.join(tmp, "per.json")))
    # utterance 0: exact; every other: one substitution (AA for AE)
    assert (rec["sub"], rec["del"], rec["ins"]) == (2702, 0, 0)
    assert rec["reference_phones"] == 2 + 2702
    assert rec["per"] == pytest.approx(2702 / 2704)
    raw = json.load(open(os.path.join(tmp, "raw.json")))
    hyps = json.load(open(os.path.join(tmp, "hyps.json")))
    assert raw[tags[0]] == ["SIL", "AA", "AE", "SIL"] and hyps[tags[0]] == ["AA", "AE"]
    assert raw[tags[1]] == ["SIL", "AA"] and hyps[tags[1]] == ["AA"]
    assert rec["phones_emitted"] == 2 + 2702 and rec["raw_emitted"] == 4 + 2 * 2702
    assert rec["distinct_strings"] == 2
    assert open(os.path.join(tmp, "per.txt")).read().startswith(f"dev-clean PER={2702 / 2704:.6f} S=2702")


def test_posterior_forward_config_builds():
    from i6_experiments.users.wu.experiments.unsupervised_asr.analysis.posterior import (
        POSTERIOR_BATCH_SIZE_FRAMES, POSTERIOR_MAX_SEQS, build_posterior_forward_config)

    net_args = dict(in_dim=1024, n_out=40, kernel=9, stride=3, n_layers=1, dropout=0.1, batch_norm=30.0,
                    residual=True, bias=False)
    cfg = build_posterior_forward_config(feature_hdfs=[tk.Path("/nonexistent/feats.hdf")], net_args=net_args)
    assert cfg.config["batch_size"] == POSTERIOR_BATCH_SIZE_FRAMES == 1_600_000
    assert cfg.config["max_seqs"] == POSTERIOR_MAX_SEQS == 200
    assert cfg.config["forward_data"]["class"] == "HDFDataset"
    assert cfg.post_config["backend"] == "torch"
    text = cfg.python_epilog[0].get()
    assert "analysis.posterior_steps" in text and "model.recognizer_only" in text
    assert "extern_data" in text
