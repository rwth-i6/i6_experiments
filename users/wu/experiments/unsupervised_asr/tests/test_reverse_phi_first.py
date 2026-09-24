"""Tests of ``reverse_model/phi_first.py`` (L2-1 stage 1, label-free) at graph-build level: the null
corpus permutation, the removed ``"run"`` granularity, the stage-1 config deltas and the wave's
refusal while its settings are undetermined (no job runs)."""

import numpy as np
import pytest
from sisyphus import tk
from sisyphus.hash import sis_hash_helper

from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first as pf
from i6_experiments.users.wu.experiments.unsupervised_asr.training.jobs import get_model_args


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    units = [p("units.0.hdf"), p("units.1.hdf")]
    return dict(
        train_feature_hdfs=[p("feats.0.hdf"), p("feats.1.hdf")], train_units_hdfs=units,
        train_original_hdfs=[p("orig.0.hdf"), p("orig.1.hdf")], dev_feature_hdfs=[p("feats.0.hdf"), p("feats.1.hdf")],
        dev_units_hdfs=list(units), dev_original_hdfs=[p("orig.0.hdf"), p("orig.1.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


def test_permute_units_keeps_multiset_and_is_deterministic():
    x = np.arange(50) % 7
    a = pf.permute_units(x, tag="utt/1", seed=0, granularity="unit")
    assert sorted(a.tolist()) == sorted(x.tolist()) and not np.array_equal(a, x)
    assert np.array_equal(a, pf.permute_units(x, tag="utt/1", seed=0, granularity="unit"))
    assert not np.array_equal(a, pf.permute_units(x, tag="utt/2", seed=0, granularity="unit"))
    assert pf.permute_units(np.array([4]), tag="u", seed=0, granularity="unit").tolist() == [4]


def test_permute_units_matches_model_permutation():
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_permute

    x = np.arange(30)
    perm = blankfree_permute.utterance_permutation("utt/7", 30, seed=pf.NULL_CORPUS_SEED)
    assert np.array_equal(pf.permute_units(x, tag="utt/7", seed=pf.NULL_CORPUS_SEED, granularity="unit"),
                          x[np.asarray(perm)])


def test_run_granularity_removed():
    with pytest.raises(ValueError):
        pf.permute_units(np.arange(5), tag="u", seed=0, granularity="run")
    with pytest.raises(ValueError):
        pf.PermutedUnitsHdfJob(units_hdfs=[tk.Path("/nonexistent/u.hdf")], seed=0, granularity="run")
    with pytest.raises(ValueError):
        pf.null_corpus(_data(), "run", alias=None)


def test_constants():
    assert pf.tau_schedule(4) == [4.0, 1.0, 1.0, 1.0]
    assert abs(pf.alloc_hours() - 1.65) < 1e-12
    assert abs(pf.alloc_hours(4) - 1.65) < 1e-12 and abs(pf.alloc_hours(8) - 3.3) < 1e-12
    assert abs(pf.alloc_hours(24) - 9.9) < 1e-12
    from i6_experiments.users.wu.experiments.unsupervised_asr.training.jobs import TIME_RQMT
    assert pf.alloc_hours(48) == TIME_RQMT
    assert pf.arm_name(3) == "em_s03" and pf.arm_name(1, "rerun") == "em_s01_rerun"


def test_stage1_config_deltas():
    cfg = pf.stage1_config(2, data=_data())
    args = get_model_args(cfg)
    assert args["null_recognizer"] is True and args["freeze_recognizer"] is True
    assert "reverse_duration_prior" not in args and "reverse_checkpoint_path" not in args
    assert cfg.config["random_seed"] == 2
    assert cfg.config["train"]["dataset"]["datasets"]["feats"]["random_seed_offset"] == 2
    assert cfg.config["optimizer"]["betas"] == [0.9, 0.999] and cfg.config["optimizer"]["eps"] == 1e-8

    for duration, mode in (("durinit", "init"), ("durfrz", "freeze")):
        a = get_model_args(pf.stage1_config(1, data=_data(), duration=duration))
        assert a["reverse_duration_prior_mode"] == mode
        assert a["reverse_duration_prior"].creator.__class__.__name__ == "BlankfreeDurationPriorMeanJob"

    phic = tk.Path("/nonexistent/phic.pt")
    assert get_model_args(pf.stage1_config(1, data=_data(), reverse_init=phic))["reverse_checkpoint_path"] is phic
    with pytest.raises(AssertionError):  # phi_c arms keep phi_c's own durations
        pf.stage1_config(1, data=_data(), reverse_init=phic, duration="durfrz")


def test_null_units_replace_train_and_dev():
    corpus = pf.null_corpus(_data(), alias=None)
    cfg = pf.stage1_config(1, data=_data(), units=corpus.out_hdfs)
    train_units = cfg.config["train"]["dataset"]["datasets"]["units"]["files"]
    dev_units = cfg.config["dev"]["datasets"]["units"]["files"]
    assert [p.get_path() for p in train_units] == [p.get_path() for p in corpus.out_hdfs]
    assert [p.get_path() for p in dev_units] == [p.get_path() for p in corpus.out_hdfs]


def test_wave_refuses_while_undetermined():
    with pytest.raises(AssertionError):
        pf.build_wave(_data(), phi_c_source=tk.Path("/nonexistent/ep60.pt"), alias=None)
    with pytest.raises(AssertionError):
        pf.build_wave(_data(), phi_c_source=tk.Path("/nonexistent/ep60.pt"), duration="durfrz", alias=None)


def test_wave_builds_24_arms_when_set():
    out = pf.build_wave(_data(), phi_c_source=tk.Path("/nonexistent/ep60.pt"), duration="durfrz",
                        num_subepochs=4, alias=None)
    jobs = [j for g in ("restarts", "nulls", "references") for j in out[g].values()]
    # 24 arms are 24 separate trainings: the exact reruns must not collapse into their restarts
    assert len(jobs) == 24 and len({j.job_id() for j in jobs}) == 24
    for s in pf.RERUN_SEEDS:
        rerun, restart = out["references"][pf.arm_name(s, "rerun")], out["restarts"][pf.arm_name(s)]
        assert rerun.job_id() != restart.job_id()
        rc, sc = rerun.returnn_config.config, restart.returnn_config.config
        assert rc[pf.RERUN_KEY] == 1 and pf.RERUN_KEY not in sc
        # everything else hashes the same (the optimizer dict holds a callable, so compare hashes)
        assert sis_hash_helper({k: v for k, v in rc.items() if k != pf.RERUN_KEY}) == sis_hash_helper(sc)
    assert all(j.rqmt["time"] == pf.alloc_hours(4) for j in jobs)
    sel = out["selection"]["selection"]
    assert len(sel.restarts) == 16 and len(sel.nulls) == 4 and len(sel.reruns) == 2
