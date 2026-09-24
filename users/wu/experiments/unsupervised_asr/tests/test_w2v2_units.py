"""CPU tests of w2v2.units on tiny synthetic L15-like HDFs (K=4, PCA-3)."""

from __future__ import annotations

import json
import os
import pickle

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2 import units as U

D = 8


def write_hdf(path, seqs):
    """RETURNN-layout float16 HDF (inputs / seqLengths / seqTags) of ``{tag: [T, D]}``."""
    import h5py

    tags = list(seqs)
    with h5py.File(path, "w") as fh:
        fh["inputs"] = np.concatenate([seqs[t] for t in tags]).astype(np.float16)
        fh["seqLengths"] = np.array([[len(seqs[t])] for t in tags], dtype=np.int32)
        fh["seqTags"] = np.array([t.encode() for t in tags], dtype="S")
    return tk.Path(path)


def seqs(prefix, n, rng):
    return {f"{prefix}-1-{i:04d}": rng.randn(rng.randint(5, 30), D).astype(np.float16) for i in range(n)}


def redirect(job, tmp, names):
    for attr, fn in names.items():
        setattr(job, attr, tk.Path(os.path.join(tmp, fn)))


def test_helpers():
    x = np.array([[1.0, 2.0], [3.0, 2.0]], dtype=np.float16)
    np.testing.assert_allclose(U.standardize_frames(x, [2.0, 2.0], [1.0, 0.0]), [[-1, 0], [1, 0]])
    assert U.standardize_frames(x, [0, 0], [1, 1]).dtype == np.float32
    dead, norm, ent = U.usage_stats([5, 5, 0, 0])
    assert dead == 2 and abs(ent - np.log(2)) < 1e-12 and abs(norm - 0.5) < 1e-12
    assert U.units_as_int_list(np.array([3, 1], dtype=np.int16)) == [3, 1]


def test_iter_hdf_seqs_order(tmp_path):
    rng = np.random.RandomState(0)
    a = seqs("1", 3, rng)
    got = list(U.iter_hdf_seqs([write_hdf(str(tmp_path / "a.hdf"), a)]))
    assert [t for t, _ in got] == list(a)
    for t, x in got:
        assert x.dtype == np.float16
        np.testing.assert_array_equal(x, a[t])


def test_quantize_assign_merge_pack(tmp_path):
    rng = np.random.RandomState(1)
    train, dev, extra = seqs("1", 12, rng), seqs("2", 4, rng), seqs("3", 5, rng)
    h_train = write_hdf(str(tmp_path / "train.hdf"), train)
    h_dev = write_hdf(str(tmp_path / "dev.hdf"), dev)
    h_all = write_hdf(str(tmp_path / "all.hdf"), {**train, **extra})
    allf = np.concatenate([x.astype(np.float32) for x in train.values()])
    stats = str(tmp_path / "stats.npy")
    np.save(stats, np.stack([allf.mean(0), allf.std(0)]).astype(np.float32))

    q = U.QuantizeStatesJob(states_hdfs={"train": [h_train], "dev": [h_dev]}, stats_npy=tk.Path(stats),
                            num_clusters=4, pca_dim=3, seed=42, max_fit_vectors=100)
    redirect(q, str(tmp_path), {"out_units": "q_units.pkl", "out_quantizer": "quantizer.pkl", "out_stats": "q.txt"})
    q.run()
    qu = pickle.load(open(tmp_path / "q_units.pkl", "rb"))
    assert list(qu) == list(train) + list(dev)
    assert all(len(qu[t]) == len(x) for t, x in {**train, **dev}.items())
    lines = open(tmp_path / "q.txt").read().splitlines()
    assert lines[0].startswith("av-state units: K=4 pca_dim=3") and "frames_used=100" in lines[1]

    # assign-only with the fixed codebook reproduces the fit-time codes
    ids_train = str(tmp_path / "ids_train.json")
    json.dump(sorted({**train, **extra}), open(ids_train, "w"))
    a = U.AssignUnitsJob(states_hdfs=[h_all], quantizer_pkl=tk.Path(str(tmp_path / "quantizer.pkl")),
                         split_ids=[tk.Path(ids_train)])
    redirect(a, str(tmp_path), {"out_units": "a_units.pkl", "out_stats": "a.txt"})
    a.run()
    au = pickle.load(open(tmp_path / "a_units.pkl", "rb"))
    for t in train:
        assert au[t].dtype == np.int16 and au[t].tolist() == qu[t]

    ids_dev = str(tmp_path / "ids_dev.json")
    json.dump(sorted(dev), open(ids_dev, "w"))
    m = U.MergeUnitsPklJob(sources=[tk.Path(str(tmp_path / "q_units.pkl")), tk.Path(str(tmp_path / "a_units.pkl"))],
                           split_ids=[tk.Path(ids_train), tk.Path(ids_dev)])
    redirect(m, str(tmp_path), {"out_units": "m_units.pkl", "out_stats": "m.txt"})
    m.run()
    mu = pickle.load(open(tmp_path / "m_units.pkl", "rb"))
    assert set(mu) == set(train) | set(dev) | set(extra)
    assert "utt-identical 12/12, frame agreement 1.000000" in open(tmp_path / "m.txt").read()

    p = U.PackUnitsJob(units_pkl=tk.Path(str(tmp_path / "m_units.pkl")))
    p.out_store = tk.Path(str(tmp_path / "store"))
    p.out_stats = tk.Path(str(tmp_path / "p.txt"))
    p.run()
    data, uids, offs, lens = (np.load(tmp_path / "store" / f) for f in U._STORE_FILES)
    assert [u.decode() for u in uids] == sorted(mu)
    for k, u in enumerate(uids):
        np.testing.assert_array_equal(data[offs[k]:offs[k] + lens[k]], mu[u.decode()])

    # coverage contract
    json.dump(sorted(train), open(ids_train, "w"))
    a2 = U.AssignUnitsJob(states_hdfs=[h_all], quantizer_pkl=tk.Path(str(tmp_path / "quantizer.pkl")),
                          split_ids=[tk.Path(ids_train)])
    redirect(a2, str(tmp_path), {"out_units": "a2.pkl", "out_stats": "a2.txt"})
    with pytest.raises(AssertionError, match="assigned keys != split ids"):
        a2.run()


def test_removed_flags_raise():
    h = {"train": [tk.Path("/x/a.hdf")], "dev": [tk.Path("/x/b.hdf")]}
    for kw in ({"dedup": True}, {"per_utt_standardize": True}):
        with pytest.raises(ValueError):
            U.QuantizeStatesJob(states_hdfs=h, stats_npy=tk.Path("/x/s.npy"), **kw)
    with pytest.raises(ValueError):
        U.QuantizeStatesJob(states_hdfs=h, stats_npy=tk.Path("/x/s.npy"), splits=("train",))
    with pytest.raises(AssertionError):
        U.AssignUnitsJob(states_hdfs=[tk.Path("/x/a.hdf")], quantizer_pkl=tk.Path("/x/q.pkl"), dedup=True)
