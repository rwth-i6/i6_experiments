"""Tests of ``reverse_model/duration_prior.py`` (label-free): the mean formula, the frame totals on
tiny HDFs and the law the model will build (phone rows at mean m, SIL uniform)."""

import h5py
import numpy as np
import pytest

from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import duration_prior as D


def _hdf(path, tags, lengths, values):
    with h5py.File(path, "w") as h:
        h.create_dataset("seqTags", data=np.array([t.encode() for t in tags], dtype="S"))
        h.create_dataset("seqLengths", data=np.array([[n] for n in lengths], dtype=np.int32))
        h.create_dataset("inputs", data=np.asarray(values, dtype=np.int32))


def test_prior_mean_frames():
    assert D.prior_mean_frames(retained_frames=80, original_frames=100, rho_hz=10.0, frame_rate_hz=50.0) == 4.0
    with pytest.raises(AssertionError):
        D.prior_mean_frames(retained_frames=120, original_frames=100, rho_hz=10.0, frame_rate_hz=50.0)


def test_build_prior_summary(tmp_path):
    tags, retained, original = ["u1", "u2", "u3"], [30, 40, 50], [40, 50, 70]
    units = tmp_path / "units.hdf"
    orig = tmp_path / "orig.hdf"
    _hdf(units, tags, retained, np.zeros(sum(retained)))
    _hdf(orig, tags, [1, 1, 1], original)
    s = D.build_prior_summary(units_hdfs=[str(units)], orig_length_hdfs=[str(orig)], rho_hz=9.6619373279,
                              frame_rate_hz=50.0)
    m = (50.0 / 9.6619373279) * (120 / 160)
    assert s["n_seqs"] == 3 and s["retained_frames"] == 120 and s["original_frames"] == 160
    assert abs(s["mean_frames"] - m) < 1e-12
    for p, row in s["law"].items():
        if p != "SIL":
            assert abs(row["mean"] - m) < 1e-6, p
    d_min, d_sil = s["reverse_config"]["d_min"], s["reverse_config"]["d_max_sil"]
    assert abs(s["law"]["SIL"]["mean"] - (d_min + d_sil) / 2) < 1e-9  # uniform on [d_min, D_SIL]


def test_frame_totals_refuse_mismatched_tags(tmp_path):
    units = tmp_path / "units.hdf"
    orig = tmp_path / "orig.hdf"
    _hdf(units, ["u1"], [3], np.zeros(3))
    _hdf(orig, ["u2"], [1], [5])
    with pytest.raises(AssertionError):
        D.stream_frame_totals([str(units)], [str(orig)])
