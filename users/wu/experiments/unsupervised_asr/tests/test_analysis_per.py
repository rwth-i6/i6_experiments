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


# ===================================================================================================
# Priority-2 tests (test plan 2026-09-24, section 4): T2.7 the edit-count property, T2.6 the
# posterior dump -> PER chain through a RETURNN CPU forward.
# ===================================================================================================

import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import ladder as LADDER  # noqa: E402


def _levenshtein(a, b) -> int:
    """Unit-cost edit distance, written independently of the package (full table)."""
    d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        d[i][0] = i
    for j in range(len(b) + 1):
        d[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
    return d[len(a)][len(b)]


def test_t2_7_edit_counts_property():
    """500 random pairs: S + D + I = Levenshtein distance, len(hyp) = len(ref) - D + I, and the
    PER reader and the ladder's copy return the same triple."""
    rng = np.random.RandomState(27)
    alphabet = ["AA", "B", "K", "IY", "SIL"]
    for n in range(500):
        ref = [alphabet[k] for k in rng.randint(0, 3 + n % 3, size=rng.randint(0, 13))]
        hyp = [alphabet[k] for k in rng.randint(0, 3 + (n // 3) % 3, size=rng.randint(0, 13))]
        s, d, i = P.edit_counts(hyp, ref)
        assert min(s, d, i) >= 0
        assert s + d + i == _levenshtein(ref, hyp), (hyp, ref, (s, d, i))
        assert len(hyp) == len(ref) - d + i, (hyp, ref, (s, d, i))
        assert LADDER.edit_counts(hyp, ref) == (s, d, i)


# --- T2.6 posterior dump -> PER --------------------------------------------------------------------

def _returnn_root() -> str:
    import returnn

    return os.path.dirname(os.path.dirname(os.path.abspath(returnn.__file__)))  # the dir holding rnn.py


T26_NET = dict(in_dim=16, n_out=40, kernel=9, stride=3, n_layers=1, dropout=0.1, batch_norm=30.0,
               residual=True, bias=False)
N_DEV_CLEAN = 2703


def _returnn_hdf(path, rows, dim, dtype):
    """``{tag: [T, dim]}`` through RETURNN's own writer (what HDFDataset reads)."""
    from returnn.datasets.hdf import SimpleHDFWriter

    writer = SimpleHDFWriter(filename=path, dim=dim, ndim=2)
    for tag, x in rows.items():
        x = np.asarray(x, dtype=dtype)
        writer.insert_batch(x[None], [x.shape[0]], [tag])
    writer.close()
    return tk.Path(path)


def test_t2_6_posterior_dump_to_per_chain(tmp_path):
    """A RETURNN CPU forward of the posterior-dump config (``analysis.posterior``) over 2703 tiny
    utterances: every dumped row = ``ConvRecognizer.eval()`` of that utterance, T_out = ceil(S / 3),
    40 columns; ``BlankfreeGreedyPerJob``'s PER = the direct argmax / collapse / drop-SIL / Levenshtein
    PER of the dump.  About 8 s on CPU (RETURNN start-up included), so not marked slow."""
    from i6_core.returnn.forward import ReturnnForwardJobV2
    from i6_core.returnn.training import PtCheckpoint

    from i6_experiments.users.wu.experiments.unsupervised_asr.analysis.posterior import (
        build_posterior_forward_config)
    from i6_experiments.users.wu.experiments.unsupervised_asr.model.recognizer import ConvRecognizer

    tmp = str(tmp_path)
    torch.manual_seed(0)
    model = ConvRecognizer(**T26_NET)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # eval must differ from train: non-default BN statistics
        model.bn.running_mean.copy_(torch.randn(16, generator=g))
        model.bn.running_var.copy_(torch.rand(16, generator=g) + 0.5)
        model.bn.weight.fill_(1.0)
        model.conv.weight.mul_(20.0)
    ckpt = os.path.join(tmp, "theta.pt")
    torch.save({"model": model.state_dict(), "epoch": 1, "step": 0}, ckpt)

    rng = np.random.RandomState(26)
    tags = [f"{1000 + k // 10}-1-{k:04d}" for k in range(N_DEV_CLEAN)]
    feats = {t: rng.randn(int(rng.randint(1, 16)), 16).astype(np.float16) for t in tags}
    orig = {t: np.array([[len(feats[t]) + int(rng.randint(0, 5))]]) for t in tags}
    phones39 = [p for p in PHONES if p != "SIL"]
    gold = {t: [phones39[k] for k in rng.randint(0, 39, size=rng.randint(1, 5))] for t in tags}
    json.dump({"dev-clean": gold}, open(os.path.join(tmp, "gold.json"), "w"))
    feat_hdf = _returnn_hdf(os.path.join(tmp, "feats.hdf"), feats, 16, "float16")
    orig_hdf = _returnn_hdf(os.path.join(tmp, "orig.hdf"), orig, 1, "int32")

    cfg = build_posterior_forward_config(feature_hdfs=[feat_hdf], net_args=T26_NET)
    cfg = ReturnnForwardJobV2.create_returnn_config(
        model_checkpoint=PtCheckpoint(tk.Path(ckpt)), returnn_config=cfg, log_verbosity=3, device="cpu")
    cfg_path = os.path.join(tmp, "returnn.config")
    cfg.write(cfg_path)
    env = dict(os.environ)  # the caller's PYTHONPATH (recipe, returnn, sisyphus) is inherited
    run = subprocess.run([sys.executable, os.path.join(_returnn_root(), "rnn.py"), cfg_path], cwd=tmp, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    assert run.returncode == 0, run.stdout[-3000:]
    frames = json.load(open(os.path.join(tmp, "posteriors.frames.json")))
    dumped = P._hdf_sequences([tk.Path(os.path.join(tmp, "posteriors.hdf"))])
    assert set(dumped) == set(frames) == set(tags)

    model.eval()
    worst, top = 0.0, 0.0
    direct_hyps = {}
    with torch.no_grad():
        for t in tags:
            x = torch.as_tensor(feats[t].astype(np.float32))[None]
            ref = model(x, torch.tensor([len(feats[t])]))[0].numpy()
            row = dumped[t]
            assert row.shape == (math.ceil(len(feats[t]) / 3), 40) == ref.shape, (t, row.shape)
            assert frames[t] == len(row)
            # the dump is a BATCHED float32 forward, the reference a single-utterance one: float32
            # rounding of a different GEMM order, so not bitwise.  The log-softmax error scales with
            # the frame's largest logit (its logsumexp), so the deviation is taken relative to the
            # frame's max |log q| (plan section 0: float32 1e-5 relative; 1e-6 asserted)
            scale = np.maximum(1.0, np.abs(ref).max(axis=1, keepdims=True))
            worst = max(worst, float((np.abs(row - ref) / scale).max()))
            top = max(top, float(np.abs(ref).max()))
            assert np.array_equal(np.argmax(row, axis=1), np.argmax(ref, axis=1)), t
            ids = np.argmax(row, axis=1)  # the direct numpy read of the dump
            ids = [int(k) for j, k in enumerate(ids) if j == 0 or k != ids[j - 1]]
            direct_hyps[t] = [PHONES[k] for k in ids if PHONES[k] != "SIL"]
    print(f"T2.6 dump vs ConvRecognizer.eval(): max relative |diff| = {worst:.3e} "
          f"(max |log q| {top:.1f}) over {len(tags)} utterances")
    assert worst <= 1e-6

    job = P.BlankfreeGreedyPerJob(posteriors=tk.Path(os.path.join(tmp, "posteriors.hdf")), features=[feat_hdf],
                                  originals=[orig_hdf], gold=tk.Path(os.path.join(tmp, "gold.json")),
                                  split="dev-clean")
    _outputs(job, tmp, {"out_per": "per.json", "out_report": "per.txt", "out_hyps": "hyps.json",
                        "out_raw_hyps": "raw.json", "out_stats": "stats.json"})
    job.run()
    rec = json.load(open(os.path.join(tmp, "per.json")))
    errors = sum(_levenshtein(gold[t], direct_hyps[t]) for t in tags)
    n_ref = sum(len(gold[t]) for t in tags)
    print(f"T2.6 PER job {rec['per']!r} direct {errors / n_ref!r} (errors {errors} / {n_ref})")
    assert rec["per"] == errors / n_ref and rec["reference_phones"] == n_ref
    assert json.load(open(os.path.join(tmp, "hyps.json"))) == direct_hyps
