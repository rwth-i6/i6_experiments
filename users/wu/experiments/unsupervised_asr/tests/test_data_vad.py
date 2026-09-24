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


# ---------------------------------------------------------------------------------------------------
# T3.1 (test plan 2026-09-24): the rVAD silence rule itself, with rVADfast replaced by a fake whose
# 10 ms labels are chosen by hand, and the pad / truncate / index rule of prepare_blankfree_data.
# ---------------------------------------------------------------------------------------------------
from i6_experiments.users.wu.experiments.unsupervised_asr.data import vad_port as VP  # noqa: E402


class _FakeVad:
    """Stands in for an ``rVADfast`` instance: ``vad(wav, sr) -> (labels, timestamps)``."""

    def __init__(self, labels_for):
        self.labels_for = labels_for
        self.calls = []

    def __call__(self, wav, sr):
        self.calls.append((wav.dtype, len(wav), sr))
        return np.asarray(self.labels_for(wav)), None


def test_rvad_silence_rule_subframes_2():
    """50 Hz (2 subframes): a frame is silence iff BOTH 10 ms labels are non-speech; tail dropped."""
    labels = [1, 1, 1, 0, 0, 1, 0, 0, 0]  # 4 whole frames + 1 dangling label
    vad = _FakeVad(lambda wav: labels)
    got = VP.rvad_silence(np.zeros(900, dtype=np.float64), sr=16000, vad=vad, subframes=2)
    assert got.dtype == bool
    np.testing.assert_array_equal(got, [False, False, False, True])
    assert vad.calls == [(np.float32, 900, 16000)]  # the waveform reaches rVADfast as float32
    # float labels (0./1.) follow the same rule
    vad = _FakeVad(lambda wav: np.asarray(labels, dtype=np.float64))
    np.testing.assert_array_equal(VP.rvad_silence(np.zeros(9), vad=vad, subframes=2), [False, False, False, True])


def test_rvad_silence_rule_subframes_4():
    """25 Hz (4 subframes): silence iff MORE than 50 % of the labels are non-speech (2 of 4 is speech)."""
    labels = [0, 0, 0, 1,  0, 0, 1, 1,  0, 1, 1, 1,  1, 1, 1, 1,  0, 0, 0, 0,  1, 0]
    vad = _FakeVad(lambda wav: labels)
    got = VP.rvad_silence(np.zeros(10), vad=vad, subframes=4)
    np.testing.assert_array_equal(got, [True, False, False, False, True])
    assert VP.SUBFRAMES == 4
    np.testing.assert_array_equal(VP.rvad_silence(np.zeros(10), vad=vad), got)  # the default is 4


def test_rvad_silence_shorter_than_one_frame():
    vad = _FakeVad(lambda wav: [1])
    got = VP.rvad_silence(np.zeros(10), vad=vad, subframes=2)
    assert got.dtype == bool and got.shape == (0,)


def _distinct_inputs(tmp_path):
    """Like ``setup_inputs`` but with three DIFFERENT waveform lengths, so a fake VAD can key its
    labels on ``len(wav)``."""
    rng = np.random.RandomState(1)
    zips, feats, units, seqs = {}, {}, {}, {}
    for split, corpus, utts, secs in (("train", "train-clean-100", ["19-198-0000", "19-198-0001"], (1.0, 1.3)),
                                      ("dev-other", "dev-other", ["116-288045-0000"], (0.8,))):
        d = tmp_path / split
        d.mkdir()
        path, _ = make_zip(d, corpus=corpus, utts=utts, seconds=secs)
        zips[split] = [path]
        split_seqs = {u: rng.randn(n_frames(len(w)), 1024).astype(np.float16) for u, w in OZ.iter_ogg_zip_audio(path)}
        hdf = str(d / "feats.hdf")
        with h5py.File(hdf, "w") as fh:
            fh["inputs"] = np.concatenate(list(split_seqs.values()))
            fh["seqLengths"] = np.array([[len(x)] for x in split_seqs.values()], dtype=np.int32)
            fh["seqTags"] = np.array([u.encode() for u in split_seqs], dtype="S")
        feats[split] = [hdf]
        units.update({u: rng.randint(0, 500, size=len(x)) for u, x in split_seqs.items()})
        seqs.update(split_seqs)
    pack_units_store(units, str(tmp_path / "store"))
    wav_len = {}
    for split, (path,) in zips.items():
        for u, w in OZ.iter_ogg_zip_audio(path):
            wav_len[u] = len(w)
    assert len(set(wav_len.values())) == len(wav_len)
    return zips, feats, units, seqs, wav_len


def test_prepare_blankfree_data_silence_rule_stubbed(tmp_path, monkeypatch):
    """T3.1: with rVADfast stubbed, a mask SHORTER than the feature stream is padded as silence, a
    LONGER one is truncated, and the kept indices are ``flatnonzero(~mask[:original])``."""
    import sys
    import types

    zips, feats, units, seqs, wav_len = _distinct_inputs(tmp_path)
    delta = {"19-198-0000": -3, "19-198-0001": +4, "116-288045-0000": 0}  # mask length - original
    rng = np.random.RandomState(7)
    masks = {}
    for u, d in delta.items():
        n = len(seqs[u]) + d
        m = rng.rand(n) < 0.4  # True = silence
        m[:2] = False  # at least two kept frames
        masks[u] = m
    by_len = {wav_len[u]: masks[u] for u in masks}

    def labels_for(wav):
        m = by_len[len(wav)]
        # 2 labels per 50 Hz frame; a silent frame gets (0, 0), a speech frame (1, 0) (one
        # non-speech label is NOT enough for silence), plus one dangling label (dropped)
        lab = np.stack([np.where(m, 0, 1), np.zeros(len(m), dtype=int)], axis=1).reshape(-1)
        return np.concatenate([lab, [0]])

    made = []

    class FakeRVADfast:
        def __init__(self, vad_threshold):
            made.append(vad_threshold)

        def __call__(self, wav, sr):
            assert sr == 16000 and wav.dtype == np.float32
            return labels_for(wav), None

    monkeypatch.setitem(sys.modules, "rVADfast", types.SimpleNamespace(rVADfast=FakeRVADfast))
    out = outputs(tmp_path / "stub", feats)
    V.prepare_blankfree_data(ogg_zips=zips, feature_hdfs=feats, units_store=str(tmp_path / "store"), **out)
    assert made == [0.4]

    expect = {}
    for u, m in masks.items():
        orig = len(seqs[u])
        full = np.pad(m, (0, max(orig - len(m), 0)), constant_values=True)[:orig]
        expect[u] = np.flatnonzero(~full)
    # the padded tail really is dropped and the truncated tail really is ignored
    assert expect["19-198-0000"][-1] < len(seqs["19-198-0000"]) - 3
    assert expect["19-198-0001"][-1] < len(seqs["19-198-0001"])

    summary = json.load(open(out["manifest_path"]))["summary"]
    for split in feats:
        idx = read(out["out_raw_index_hdfs"][split][0])
        f = read(out["out_feature_hdfs"][split][0])
        un = read(out["out_units_hdfs"][split][0])
        for u, ind in idx.items():
            np.testing.assert_array_equal(ind.reshape(-1), expect[u])
            np.testing.assert_array_equal(f[u], seqs[u][expect[u]])
            np.testing.assert_array_equal(un[u].reshape(-1), units[u][expect[u]])
        assert summary[split]["kept_frames"] == sum(len(expect[u]) for u in idx)
        assert summary[split]["original_frames"] == sum(len(seqs[u]) for u in idx)
