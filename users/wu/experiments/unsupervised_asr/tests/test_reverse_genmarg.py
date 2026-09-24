"""Tests of ``reverse_model/genmarg.py`` (the label-free L2-1 reads and selection) without running a
job: the selection rule on synthetic records, the bed helper, the cut options and the forward jobs'
resources."""

import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg as G
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first as pf

TAGS = ["a", "b", "c"]


def _rec(value, shuffle=None, impossible=()):
    return {"dataset": "cv_holdout", "settings": {"shuffle_seed": shuffle},
            "summary": {"expected_nonsil_rate": 1.0, "n_impossible": len(impossible)},
            "per_utterance": {t: {"nll_tau1_per_frame": value, "impossible": t in impossible,
                                  "expected_nonsil_tokens": 10.0, "original_frames": 50} for t in TAGS}}


def _dec(rate):
    return {"dataset": "cv_holdout", "settings": {"shuffle_seed": None},
            "per_utterance": {t: {} for t in TAGS}, "emitted_nonsil_rate": {"pooled_hz": rate, "n": 3}}


def _inputs(null_values=(6.0, 6.05, 6.1, 6.02), rates=None):
    restarts = {pf.arm_name(s): _rec(5.0 + 0.01 * s) for s in range(1, 17)}
    rates = rates or {k: 10.0 for k in restarts}
    return dict(
        restarts=restarts,
        nulls={pf.arm_name(s, "null"): _rec(v, shuffle=0) for s, v in enumerate(null_values, 1)},
        nulls_real={pf.arm_name(s, "null"): _rec(7.0) for s in range(1, 5)},
        phi_c={pf.arm_name(s, "phic"): _rec(5.5) for s in (1, 2)},
        reruns={"em_s01": _rec(5.01 + 0.004), "em_s02": _rec(5.02)},
        restart_decodes={k: _dec(r) for k, r in rates.items()},
    )


def test_decide_signal_with_void_restart():
    kw = _inputs()
    kw["restart_decodes"]["em_s01"] = _dec(20.0)  # outside [5.80, 14.49]: VOID
    out = G.GenMargSelectionJob.decide(**kw)
    assert out["void"]["em_s01"] and out["eligible"] == 15
    assert out["selected"] == "em_s02"
    assert out["best_null"] == "null_s01" and abs(out["null_range"] - 0.1) < 1e-12
    assert abs(out["identity_band"] - 0.004) < 1e-12 and abs(out["margin"] - 0.1) < 1e-12
    assert abs(out["gap"] - (6.0 - 5.02)) < 1e-12 and out["verdict"] == "SIGNAL"
    assert out["reference_reading"] == "BEYOND PRIVATE CODE"


def test_decide_no_signal_and_no_eligible():
    out = G.GenMargSelectionJob.decide(**_inputs(null_values=(5.02, 5.06, 5.07, 5.08)))
    assert out["selected"] == "em_s01" and out["verdict"] == "NO SIGNAL"
    kw = _inputs()
    kw["restart_decodes"] = {k: _dec(1.0) for k in kw["restarts"]}
    out = G.GenMargSelectionJob.decide(**kw)
    assert out["selected"] is None and out["verdict"] == "NO ELIGIBLE RESTART"


def test_decide_refuses_a_real_null():
    kw = _inputs()
    kw["nulls"]["null_s01"] = _rec(6.0)  # nulls are gated on the permuted holdout
    with pytest.raises(AssertionError):
        G.GenMargSelectionJob.decide(**kw)


def test_decide_pairing_set_drops_impossible():
    kw = _inputs()
    kw["nulls"]["null_s02"] = _rec(6.05, shuffle=0, impossible=("c",))
    out = G.GenMargSelectionJob.decide(**kw)
    assert out["pairing_utterances"] == 2 and out["utterances"] == 3


def test_select_utterances():
    tags = [f"t{i}" for i in range(20)]
    a = G.select_utterances(tags, 5)
    assert a == G.select_utterances(list(reversed(tags)), 5) and len(set(a)) == 5
    assert G.GenMargSampleJob.select(tags, None, 0) == sorted(tags)


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


def test_reads_bed_and_forward_jobs():
    reads = pf.reads_bed(_data())
    bed = reads["bed"]
    assert list(bed["model_args"]) == list(G.BED_MODEL_ARG_KEYS)
    assert bed["model_args"]["temperature_schedule"] == list(G.READ_TEMPERATURE_SCHEDULE)
    assert "recognizer_checkpoint_path" not in bed["model_args"]
    jobs = G.genmarg_reads(tk.Path("/nonexistent/phi.pt"), "x", alias=None, **reads)["cv_holdout"]
    assert set(jobs) == {"sample", "marginal", "decode"}
    assert jobs["sample"].n is None
    assert jobs["marginal"].rqmt == {"gpu": 1, "cpu": 4, "mem": 32, "time": 2}


def test_cut_options_raise():
    reads = pf.reads_bed(_data())
    phi = tk.Path("/nonexistent/phi.pt")
    for bad in ({"gap": True}, {"report": True}, {"datasets": ("dev-other",)}):
        with pytest.raises(ValueError):
            G.genmarg_reads(phi, "x", alias=None, **reads, **bad)
    with pytest.raises(AssertionError):
        G.genmarg_reads("/nonexistent/phi.pt", "x", alias=None, **reads)
