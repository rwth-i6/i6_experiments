"""CPU tests of analysis.gaps (items -> reverse-score forward -> readers) on tiny synthetic inputs.

The last test runs the real RETURNN forward (``rnn.py`` on CPU) on a randomly initialised reverse
model and checks every score against a direct ``reverse.evaluate`` call on the same items; it is
skipped when ``rnn.py`` is not next to the importable ``returnn`` package.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import h5py
import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.analysis import gaps as G
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import PHONE2ID, SIL_ID


def test_s1a_helpers():
    assert G.with_edge_sil([3, 4]) == [SIL_ID, 3, 4, SIL_ID]
    tags = [f"t{k}" for k in range(10)]
    sel = G.select_utterances(list(reversed(tags)), 4, seed=0)
    perm = np.random.RandomState(0).permutation(10)
    assert sel == [sorted(tags)[i] for i in perm[:4]]
    assert G.speaker_of("116-288045-0009") == "116"
    clusters = G._clusters_by_speaker(["2-1-0", "1-1-0", "2-1-1", "1-1-1", "3-1-0"])
    assert [c.tolist() for c in clusters] == [[1, 3], [0, 2], [4]]
    entries = [("a", "s", 5, 10), ("b", "s", 7, 10), ("c", "s", 9, 10), ("d", "x", 3, 10)]
    pairing = G.build_derangement(entries)
    assert pairing == {"a": "b", "b": "a", "c": "b"}  # d has no same-speaker donor; ties by tag
    pairing = G.build_derangement(entries, is_feasible=lambda t, donor: donor != "b")
    assert pairing == {"a": "c", "b": "a", "c": "a"}


def _write_hdf(path, rows, dtype):
    tags = list(rows)
    data = np.concatenate([np.asarray(rows[t], dtype=dtype) for t in tags], axis=0)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("inputs", data=data)
        fh.create_dataset("seqLengths", data=np.array([[len(rows[t]), 0] for t in tags], dtype="int32"))
        fh.create_dataset("seqTags", data=np.array([t.encode() for t in tags]))
    return tk.Path(path)


GOLD = {"10-1-0": ["AA", "B"], "10-1-1": ["AE", "K", "S"], "10-1-2": ["AA"],
        "20-1-0": ["IY", "T"], "20-1-1": ["IY", "T", "S", "AA"], "20-1-2": []}


def _inputs(tmp):
    rng = np.random.RandomState(0)
    raw = {t: ["SIL"] + (g or ["AA"]) + ["SIL"] for t, g in GOLD.items()}
    units = {t: rng.randint(0, 500, size=4 * (len(g) + 2)) for t, g in GOLD.items()}
    tags = sorted(GOLD)
    np.savez(os.path.join(tmp, "eta.npz"), eta=rng.randn(len(tags), 16).astype(np.float32),
             tags=np.array(tags))
    json.dump(raw, open(os.path.join(tmp, "raw.json"), "w"))
    json.dump({"tiny": GOLD}, open(os.path.join(tmp, "gold.json"), "w"))
    return dict(raw_hyps=tk.Path(os.path.join(tmp, "raw.json")), gold=tk.Path(os.path.join(tmp, "gold.json")),
                units=[_write_hdf(os.path.join(tmp, "units.hdf"), units, "int32")],
                eta_npz=tk.Path(os.path.join(tmp, "eta.npz")), split="tiny")


def _items(tmp, with_gold):
    d = os.path.join(tmp, "gold" if with_gold else "dec")
    os.makedirs(d, exist_ok=True)
    job = G.ReverseGapItemsJob(**_inputs(tmp), with_gold=with_gold)
    job.out_items, job.out_meta = tk.Path(os.path.join(d, "items.hdf")), tk.Path(os.path.join(d, "items.json"))
    job.run()
    return job, json.load(open(os.path.join(d, "items.json")))


def test_items_job_selection_pairing_and_hdf(tmp_path):
    tmp = str(tmp_path)
    _, dec = _items(tmp, False)
    job, meta = _items(tmp, True)
    # the empty-gold utterance is never selected; every other one is, and every decode is feasible
    assert meta["selected"] == 5 and meta["decoded_feasible"] == 5 and "20-1-2" not in meta["rows"]
    for key in ("selected", "decoded_feasible", "pairing_decoded", "tags_decoded"):
        assert dec[key] == meta[key], key
    assert all(meta["pairing_decoded"][t] != t for t in meta["pairing_decoded"])
    assert all(G.speaker_of(meta["pairing_gold"][t]) == G.speaker_of(t) for t in meta["pairing_gold"])
    assert list(meta["conditions"]) == list(G.GAP_CONDITIONS) and list(dec["conditions"]) == list(G.GAP_CONDITIONS[:2])
    assert meta["rows"]["10-1-1"]["n_gold"] == 5  # edge SIL on both ends
    with h5py.File(job.out_items.get_path(), "r") as fh:
        tags = [t.decode() for t in fh["seqTags"][:]]
        assert len(tags) == meta["n_items"] == sum(len(v) for v in meta["conditions"].values())
        assert tags[0] == f"decoded_own/{meta['conditions']['decoded_own'][0]}"
        phones = fh["targets/data/phones"][:]
        index = fh["targets/data/index"][:].reshape(-1, 3)
    # first gold_deranged item carries its DONOR's gold string
    k = tags.index(f"gold_deranged/{meta['conditions']['gold_deranged'][0]}")
    assert index[k].tolist() == [3, 0, len(meta["conditions"]["gold_deranged"])]
    assert phones.max() < 40 and index[:, 0].max() == 3


def test_readers_on_synthetic_scores(tmp_path):
    tmp = str(tmp_path)
    _, meta = _items(tmp, True)
    rows = meta["rows"]
    scores = {}
    for name, tags in meta["conditions"].items():
        for t in tags:
            base = -3.0 * rows[t]["frames"]
            scores[f"{name}/{t}"] = base + {"decoded_own": 1.0, "decoded_deranged": -2.0,
                                            "gold_own": 0.5, "gold_deranged": -1.0}[name]
    json.dump(scores, open(os.path.join(tmp, "scores.json"), "w"))

    dj = G.BlankfreeDerangementGapJob(items=tk.Path(os.path.join(tmp, "gold", "items.json")),
                                      scores=tk.Path(os.path.join(tmp, "scores.json")))
    dj.out_summary, dj.out_report = (tk.Path(os.path.join(tmp, "dg.json")), tk.Path(os.path.join(tmp, "dg.txt")))
    dj.run()
    rec = json.load(open(os.path.join(tmp, "dg.json")))
    frames = sum(rows[t]["frames"] for t in meta["tags_decoded"])
    assert rec["gap"] == pytest.approx(3.0 * len(meta["tags_decoded"]) / frames)
    assert rec["masked_frames"] == frames and rec["matched_utterances"] == len(meta["tags_decoded"])

    j4 = G.BlankfreeDecodeGapJob(items=tk.Path(os.path.join(tmp, "gold", "items.json")),
                                 scores=tk.Path(os.path.join(tmp, "scores.json")),
                                 reference_gap=tk.Path(os.path.join(tmp, "dg.json")), name="tiny", n_boot=50)
    j4.out_summary = tk.Path(os.path.join(tmp, "d4.json"))
    j4.out_per_utterance = tk.Path(os.path.join(tmp, "d4_utt.json"))
    j4.out_report = tk.Path(os.path.join(tmp, "d4.txt"))
    j4.run()
    d4 = json.load(open(os.path.join(tmp, "d4.json")))
    assert d4["registered_decoded_gap_per_frame"] == pytest.approx(rec["gap"])
    c = d4["contrasts"]
    assert c["gold_minus_decoded"]["per_utterance"]["mean"] == pytest.approx(-0.5)
    assert c["gold_minus_deranged_gold"]["per_utterance"]["mean"] == pytest.approx(1.5)
    assert c["gold_minus_deranged_gold"]["per_utterance"]["ci95"] == pytest.approx([1.5, 1.5])
    assert d4["n_decoded_above_gold"] == d4["utterances"]

    # a reference gap that does not match is refused
    json.dump({**rec, "gap": rec["gap"] + 0.01}, open(os.path.join(tmp, "dg_bad.json"), "w"))
    j4.reference_gap = tk.Path(os.path.join(tmp, "dg_bad.json"))
    with pytest.raises(AssertionError, match="not scoring what"):
        j4.run()


def test_reverse_score_config_builds():
    cfg = G.build_reverse_score_config(items=tk.Path("/nonexistent/items.hdf"))
    assert cfg.config["forward_data"]["class"] == "HDFDataset"
    assert cfg.config["forward_data"]["seq_ordering"] == "default"
    assert cfg.config["batch_size"] == G.REVERSE_SCORE_BATCH_SIZE
    text = cfg.python_epilog[0].get()
    assert "reverse_score_forward_step" in text and "get_reverse_model" in text and "extern_data" in text


def test_returnn_forward_matches_direct_evaluate(tmp_path):
    """The real RETURNN forward (CPU) gives exactly ``reverse.evaluate``'s per-condition scores."""
    import returnn
    import torch

    from i6_core.returnn.forward import ReturnnForwardJobV2

    from i6_experiments.users.wu.experiments.unsupervised_asr.model.reverse import (
        ReverseConfig, SegmentalReverseModel, evaluate)

    rnn = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(returnn.__file__))), "rnn.py")
    if not os.path.exists(rnn):
        pytest.skip(f"no rnn.py next to the returnn package ({rnn})")
    tmp = str(tmp_path)
    _, meta = _items(tmp, True)
    torch.manual_seed(0)
    model = SegmentalReverseModel(ReverseConfig())
    ckpt = os.path.join(tmp, "model.pt")
    torch.save({"model": model.state_dict(), "epoch": 1, "step": 0}, ckpt)

    fwd = G.reverse_score_dump(name="test", reverse_checkpoint=tk.Path(ckpt),
                               items=tk.Path(os.path.join(tmp, "gold", "items.hdf")),
                               returnn_exe=tk.Path(sys.executable),
                               returnn_root=tk.Path(os.path.dirname(rnn)))
    cfg = ReturnnForwardJobV2.create_returnn_config(
        model_checkpoint=fwd.model_checkpoint, returnn_config=fwd.returnn_config,
        log_verbosity=fwd.log_verbosity, device=fwd.device)
    run_dir = os.path.join(tmp, "fwd")
    os.makedirs(run_dir)
    cfg.black_formatting, cfg._black_path = False, None
    cfg.write(os.path.join(run_dir, "returnn.config"))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    subprocess.check_call([sys.executable, rnn, "returnn.config"], cwd=run_dir, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    got = json.load(open(os.path.join(run_dir, "scores.json")))

    # the source's scoring: one evaluate() call per condition over its items in order
    with h5py.File(os.path.join(tmp, "gold", "items.hdf"), "r") as fh:
        tags = [t.decode() for t in fh["seqTags"][:]]
        z_all = fh["inputs"][:]
        lens = fh["seqLengths"][:]
        phones_all = fh["targets/data/phones"][:]
        eta_all = fh["targets/data/eta"][:].reshape(-1, 16)
    order = sorted(fh_key for fh_key in ("eta", "index", "phones"))
    zo = np.r_[0, np.cumsum(lens[:, 0])]
    po = np.r_[0, np.cumsum(lens[:, 1 + order.index("phones")])]
    items = {tag: (z_all[zo[i]:zo[i + 1]].tolist(), phones_all[po[i]:po[i + 1]].tolist(), eta_all[i])
             for i, tag in enumerate(tags)}
    model.eval()
    for name, cond_tags in meta["conditions"].items():
        keys = [f"{name}/{t}" for t in cond_tags]
        _, _, rows = evaluate(model, [items[k] for k in keys], per_utterance=True)
        for r in rows:
            assert got[keys[r["index"]]] == r["log_p"], keys[r["index"]]
    assert len(got) == meta["n_items"]
