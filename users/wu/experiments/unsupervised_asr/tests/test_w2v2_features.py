"""CPU tests of w2v2.features / w2v2.forward without the wav2vec2 weights: the seq-order job on a tiny
ogg zip, the RETURNN config, the statistics helpers and the feature callback."""

from __future__ import annotations

import os
import pickle
import types

import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2 import features as F
from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2 import forward as FW

from test_data_ogg_zip import UTTS, make_zip  # sibling test module (pytest prepends this dir)


def run_seq_order(tmp_path, **kw):
    job = F.L15ForwardSeqOrderJob(**kw)
    job.out_seq_list = tk.Path(str(tmp_path / "seq_list.txt"))
    job.out_seq_order = tk.Path(str(tmp_path / "seq_order.py"))
    job.out_stats = tk.Path(str(tmp_path / "stats.txt"))
    job.run()
    seq_list = open(tmp_path / "seq_list.txt").read().split()
    order = eval(open(tmp_path / "seq_order.py").read())
    return seq_list, order


def test_seq_order_job(tmp_path):
    path, _ = make_zip(tmp_path)
    z = tk.Path(path)
    full = lambda u: f"dev-other/{u}/{u}"  # noqa: E731
    seq_list, order = run_seq_order(tmp_path, ogg_zip=z, shard=(0, 1))
    assert seq_list == [full(u) for u in sorted(UTTS)] and order == {t: i for i, t in enumerate(seq_list)}
    seq_list, _ = run_seq_order(tmp_path, ogg_zip=z, shard=(1, 2))
    assert seq_list == [full(sorted(UTTS)[1])]
    ids = tmp_path / "ids.txt"
    ids.write_text("\n".join([UTTS[2], UTTS[0]]) + "\n")
    seq_list, order = run_seq_order(tmp_path, ogg_zip=z, ids=tk.Path(str(ids)), order="given")
    assert seq_list == [full(UTTS[2]), full(UTTS[0])] and order[full(UTTS[2])] == 0
    ids.write_text("nope-1-0000\n")
    with pytest.raises(AssertionError, match="not in the zip"):
        run_seq_order(tmp_path, ogg_zip=z, ids=tk.Path(str(ids)))
    for bad in ({}, {"shard": (0, 1), "ids": tk.Path(str(ids))}, {"shard": (2, 2)}, {"shard": (0, 1), "order": "x"}):
        with pytest.raises(ValueError):
            F.L15ForwardSeqOrderJob(ogg_zip=z, **bad)


def test_forward_config(tmp_path):
    cfg = F.build_l15_forward_config(ogg_zip=tk.Path("/x/out.ogg.zip"), seq_list=tk.Path("/x/l.txt"),
                                     seq_order=tk.Path("/x/o.py"), hf_model_dir=tk.Path("/x/m"), expected_num_seqs=3)
    data = cfg.config["forward_data"]
    assert data["class"] == "MetaDataset" and data["data_map"] == {FW.AUDIO_KEY: ("zip_dataset", "data")}
    zd = data["datasets"]["zip_dataset"]
    assert zd["class"] == "OggZipDataset" and zd["seq_ordering"] == "sorted"
    assert zd["audio"] == {"features": "raw", "peak_normalization": False, "preemphasis": None}
    assert cfg.config["max_seqs"] == 1
    cfg.black_formatting, cfg._black_path = False, None
    out = str(tmp_path / "returnn.config")
    cfg.write(out)
    text = open(out).read()
    assert "w2v2.forward\", fromlist=[\"get_model\"]" in text and "'expected_num_seqs': 3" in text


def test_stats_helpers():
    rng = np.random.RandomState(0)
    xs = [rng.randn(n, 6).astype(np.float32) for n in (3, 7, 1)]
    rs = FW.RunningStats(6)
    for x in xs:
        rs.add(x)
    allx = np.concatenate(xs).astype(np.float64)
    np.testing.assert_allclose(rs.stats(), np.stack([allx.mean(0), allx.std(0)]), rtol=1e-5, atol=1e-6)
    p = FW.per_utt_stats(xs[2])
    assert p.dtype == np.float32 and p.shape == (2, 6) and np.all(p[1] == np.float32(1e-5))


def test_callback_writes_bare_ids(tmp_path):
    pytest.importorskip("returnn")
    import h5py

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        cb = FW.L15FeatureCallback(feature_dim=4, expected_num_seqs=2)
        cb.init()
        rng = np.random.RandomState(0)
        xs = {"dev-other/1-2-0003/1-2-0003": rng.randn(5, 4).astype(np.float32) * 100,
              "dev-other/1-2-0001/1-2-0001": rng.randn(3, 4).astype(np.float32)}
        for tag, x in xs.items():
            cb.process_seq(seq_tag=tag, outputs={"states": types.SimpleNamespace(raw_tensor=x)})
        cb.finish()
        with h5py.File("feats.hdf", "r") as fh:
            assert [t.decode() for t in fh["seqTags"][:]] == ["1-2-0003", "1-2-0001"]
            assert fh["inputs"].dtype == np.float16
            np.testing.assert_array_equal(fh["inputs"][:5], xs["dev-other/1-2-0003/1-2-0003"].astype(np.float16))
        per = pickle.load(open("perutt_stats.pkl", "rb"))
        np.testing.assert_array_equal(per["1-2-0001"], FW.per_utt_stats(xs["dev-other/1-2-0001/1-2-0001"]))
        assert np.load("global_stats.npy").shape == (2, 4)
        with pytest.raises(ValueError):  # a bare id is not a segment name
            cb.process_seq(seq_tag="1-2-0001", outputs={"states": types.SimpleNamespace(raw_tensor=xs["dev-other/1-2-0001/1-2-0001"])})
    finally:
        os.chdir(cwd)
