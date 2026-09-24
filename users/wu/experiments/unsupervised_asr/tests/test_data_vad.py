"""CPU tests of data.vad: the joint rVAD masking end to end on tiny synthetic ogg zips, feature HDFs
and a packed unit store; the count guard and the constructor checks."""

from __future__ import annotations

import json
import os

import h5py
import numpy as np
import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.data import ogg_zip as OZ
from i6_experiments.users.wu.experiments.unsupervised_asr.data import vad as V
from i6_experiments.users.wu.experiments.unsupervised_asr.w2v2.units import pack_units_store

from test_data_ogg_zip import make_zip  # sibling test module (pytest prepends this dir)


def n_frames(n):
    for k, s in zip((10, 3, 3, 3, 3, 2, 2), (5, 2, 2, 2, 2, 2, 2)):
        n = (n - k) // s + 1
    return n


def setup_inputs(tmp_path):
    rng = np.random.RandomState(0)
    zips, feats, units = {}, {}, {}
    for split, corpus, utts in (("train", "train-clean-100", ["19-198-0000", "19-198-0001"]),
                                ("dev-other", "dev-other", ["116-288045-0000"])):
        d = tmp_path / split
        d.mkdir()
        path, _ = make_zip(d, corpus=corpus, utts=utts, seconds=(1.2, 1.5)[:len(utts)])
        zips[split] = [path]
        seqs = {u: rng.randn(n_frames(len(w)), 1024).astype(np.float16) for u, w in OZ.iter_ogg_zip_audio(path)}
        hdf = str(d / "feats.hdf")
        with h5py.File(hdf, "w") as fh:
            fh["inputs"] = np.concatenate(list(seqs.values()))
            fh["seqLengths"] = np.array([[len(x)] for x in seqs.values()], dtype=np.int32)
            fh["seqTags"] = np.array([u.encode() for u in seqs], dtype="S")
        feats[split] = [hdf]
        units.update({u: rng.randint(0, 500, size=len(x)) for u, x in seqs.items()})
        feats.setdefault("_seqs", {}).update(seqs)
    pack_units_store(units, str(tmp_path / "store"))
    return zips, feats, units


def outputs(tmp_path, feats):
    tmp_path.mkdir()
    mk = lambda kind: {s: [str(tmp_path / f"{kind}.{s}.hdf")] for s in feats if s != "_seqs"}  # noqa: E731
    return dict(out_feature_hdfs=mk("o_feats"), out_units_hdfs=mk("o_units"), out_raw_index_hdfs=mk("o_idx"),
                out_orig_length_hdfs=mk("o_len"), manifest_path=str(tmp_path / "manifest.json"))


def read(path):
    with h5py.File(path, "r") as fh:
        tags = [t.decode() for t in fh["seqTags"][:]]
        lens = fh["seqLengths"][:, 0]
        x = fh["inputs"][:]
    offs = np.concatenate([[0], np.cumsum(lens)])
    return {t: x[offs[i]:offs[i + 1]] for i, t in enumerate(tags)}


def test_prepare_blankfree_data(tmp_path):
    zips, feats, units = setup_inputs(tmp_path)
    seqs = feats.pop("_seqs")
    out = outputs(tmp_path / "r1", feats)
    V.prepare_blankfree_data(ogg_zips=zips, feature_hdfs=feats, units_store=str(tmp_path / "store"), **out)
    summary = json.load(open(out["manifest_path"]))["summary"]
    assert summary["train"]["utterances"] == 2 and summary["dev-other"]["utterances"] == 1
    assert summary["train"]["reference_kept_frames"] is None
    for split in feats:
        idx = read(out["out_raw_index_hdfs"][split][0])
        f = read(out["out_feature_hdfs"][split][0])
        un = read(out["out_units_hdfs"][split][0])
        ol = read(out["out_orig_length_hdfs"][split][0])
        for u, ind in idx.items():
            ind = ind.reshape(-1)
            assert len(ind) >= 2 and np.all(np.diff(ind) > 0) and ind[-1] < len(seqs[u])
            np.testing.assert_array_equal(f[u], seqs[u][ind])
            np.testing.assert_array_equal(un[u].reshape(-1), units[u][ind])
            assert int(ol[u].reshape(-1)[0]) == len(seqs[u])

    # the count guard: a given key that disagrees raises after the manifest is written
    out = outputs(tmp_path / "r2", feats)
    wrong = {"train": {"utterances": 3}, "dev-other": {}}
    with pytest.raises(ValueError, match="counts"):
        V.prepare_blankfree_data(ogg_zips=zips, feature_hdfs=feats, units_store=str(tmp_path / "store"),
                                 expected_counts=wrong, **out)
    assert os.path.exists(out["manifest_path"])
    right = {"train": {"utterances": 2, "original_frames": sum(len(seqs[u]) for u in ("19-198-0000", "19-198-0001"))},
             "dev-other": {"utterances": 1}}
    out = outputs(tmp_path / "r3", feats)
    V.prepare_blankfree_data(ogg_zips=zips, feature_hdfs=feats, units_store=str(tmp_path / "store"),
                             expected_counts=right, **out)


def test_job_ctor_checks():
    f = {"train": [tk.Path("/x/a.hdf")]}
    with pytest.raises(ValueError):
        V.BlankfreeVadHdfJob(ogg_zips={"dev": [tk.Path("/x/z.zip")]}, feature_hdfs=f, units_store=tk.Path("/x/s"))
    with pytest.raises(ValueError):
        V.BlankfreeVadHdfJob(ogg_zips={"train": [tk.Path("/x/z.zip")]}, feature_hdfs=f, units_store=tk.Path("/x/s"),
                             expected_counts={"train": {"frames": 1}})
    with pytest.raises(TypeError):
        V.BlankfreeVadHdfJob(ogg_zips={"train": [tk.Path("/x/z.zip")]}, feature_hdfs=f, units_store=tk.Path("/x/s"),
                             reproduction_dir=tk.Path("/x/r"))
    job = V.BlankfreeVadHdfJob(ogg_zips={"train": [tk.Path("/x/z.zip")]}, feature_hdfs=f, units_store=tk.Path("/x/s"),
                               expected_counts={"train": {"utterances": 5}})
    assert job.expected_counts == {"train": {"utterances": 5}}
