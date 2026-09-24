"""Tests of the P0 amendments (SAE_i6_P0.md, design review 2026-09-24): the ``ctrl_20_s1`` preset,
p0's dev-other PER read and the report-only VAD count check under an ffmpeg accept label.  CPU,
graph-build level plus one small VAD run; no job runs."""

from __future__ import annotations

import copy
import json
import os

import pytest
from sisyphus import gs, tk
from sisyphus.hash import sis_hash_helper

from i6_experiments.users.wu.experiments.unsupervised_asr.data import vad as V
from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms, config, jobs


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")],
        train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")],
        dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")],
        dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"),
        dev_segments=p("cv.segments"),
        prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"),
        flat_checkpoint=p("flat_init.pt"),
    )


@pytest.fixture
def private_jobs(monkeypatch):
    """A private job registry, so jobs built here do not merge with other tests' jobs."""
    import sisyphus.job

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})


# ================================================================================================
# ctrl_20_s1
# ================================================================================================
def test_ctrl_20_s1_seeds_reach_the_config(private_jobs):
    from i6_experiments.users.wu.experiments.unsupervised_asr.training.init import FlatRecognizerInitJob

    ctrl = arms.ctrl_20(data=_data())
    s1 = arms.ctrl_20_s1(data=_data())
    assert (s1.name, s1.num_epochs, s1.keep_epochs) == ("ctrl_20_s1", 20, (1, 4, 10, 20))
    assert arms.CTRL_20_S1_SEEDS == {"flat_seed": 1, "random_seed": 1, "random_seed_offset": 1000}
    assert arms.ARM_PRESETS["ctrl_20_s1"] is arms.ctrl_20_s1

    c = s1.returnn_config.config
    assert c["random_seed"] == 1
    assert c["train"]["dataset"]["datasets"]["feats"]["random_seed_offset"] == 1000
    assert "random_seed_offset" not in c["dev"]["datasets"]["feats"]
    assert "random_seed" not in ctrl.returnn_config.config
    assert "random_seed_offset" not in ctrl.returnn_config.config["train"]["dataset"]["datasets"]["feats"]

    # theta's init: the seed-1 flat init job, with the bed's NET_ARGS
    rec = jobs.get_model_args(s1.returnn_config)["recognizer_checkpoint_path"]
    flat = rec.creator
    assert isinstance(flat, FlatRecognizerInitJob) and flat.seed == 1
    assert flat.net_args == dict(config.NET_ARGS)
    assert flat is FlatRecognizerInitJob(net_args=config.NET_ARGS, seed=1)
    assert flat._sis_id() != FlatRecognizerInitJob(net_args=config.NET_ARGS, seed=0)._sis_id()
    assert _data_flat(ctrl).creator is None  # ctrl_20 keeps data["flat_checkpoint"] as given

    # nothing else differs: undo the three deltas and the hash is ctrl_20's
    undone = copy.deepcopy(s1.returnn_config)
    del undone.config["random_seed"]
    del undone.config["train"]["dataset"]["datasets"]["feats"]["random_seed_offset"]
    jobs.get_model_args(undone)["recognizer_checkpoint_path"] = jobs.get_model_args(ctrl.returnn_config)[
        "recognizer_checkpoint_path"]
    assert sis_hash_helper(undone) == sis_hash_helper(ctrl.returnn_config)
    assert sis_hash_helper(s1.returnn_config) != sis_hash_helper(ctrl.returnn_config)


def _data_flat(arm):
    return jobs.get_model_args(arm.returnn_config)["recognizer_checkpoint_path"]


def test_ctrl_20_s1_training_job_is_its_own(private_jobs):
    ctrl = jobs.train_arm(**arms.ctrl_20(data=_data()).job_kwargs())
    s1 = jobs.train_arm(**arms.ctrl_20_s1(data=_data()).job_kwargs())
    assert s1 is not ctrl and s1._sis_id() != ctrl._sis_id()
    assert s1.rqmt == ctrl.rqmt and sorted(s1.out_checkpoints) == sorted(ctrl.out_checkpoints)


# ================================================================================================
# report-only VAD counts under an ffmpeg accept label
# ================================================================================================
def _vad_job(**kw):
    f = {"train": [tk.Path("/x/a.hdf")]}
    return V.BlankfreeVadHdfJob(ogg_zips={"train": [tk.Path("/x/z.zip")]}, feature_hdfs=f,
                                units_store=tk.Path("/x/s"), expected_counts={"train": {"utterances": 5}}, **kw)


def test_vad_report_only_flag_hash(private_jobs):
    strict = _vad_job()
    assert strict.counts_report_only is False and strict.out_counts_report is None
    assert _vad_job(counts_report_only=False) is strict  # the excluded default: the strict job's hash
    report = _vad_job(counts_report_only=True)
    assert report is not strict and report._sis_id() != strict._sis_id()
    assert report.out_counts_report.get_path().endswith("counts_vs_expected.json")
    with pytest.raises(ValueError, match="counts_report_only"):
        V.BlankfreeVadHdfJob(ogg_zips={"train": [tk.Path("/x/z.zip")]}, feature_hdfs={"train": [tk.Path("/x/a.hdf")]},
                             units_store=tk.Path("/x/s"), counts_report_only=True)


def test_write_counts_report(tmp_path):
    summary = {"train": {"utterances": 10, "original_frames": 1000, "kept_frames": 795},
               "dev-other": {"utterances": 2, "original_frames": 50, "kept_frames": 40}}
    expected = {"train": {"utterances": 10, "kept_frames": 800}, "dev-other": {"kept_frames": 40}}
    path = str(tmp_path / "r.json")
    rep = V.write_counts_report(summary, expected, path)
    assert json.load(open(path)) == rep
    assert rep["all_equal"] is False
    assert rep["splits"]["train"]["kept_frames"] == {"observed": 795, "expected": 800, "rel_diff": -5 / 800}
    assert rep["splits"]["train"]["utterances"]["rel_diff"] == 0.0
    assert set(rep["splits"]["train"]) == {"utterances", "kept_frames"}  # only the given keys
    assert rep["max_abs_rel_diff"] == pytest.approx(5 / 800)


def test_prepare_blankfree_data_report_only_does_not_raise(tmp_path):
    from test_data_vad import outputs, setup_inputs  # sibling test module (pytest prepends this dir)

    zips, feats, _units = setup_inputs(tmp_path)
    feats.pop("_seqs")
    wrong = {"train": {"utterances": 3, "kept_frames": 1}, "dev-other": {"utterances": 1}}
    out = outputs(tmp_path / "strict", feats)
    with pytest.raises(ValueError, match="counts"):
        V.prepare_blankfree_data(ogg_zips=zips, feature_hdfs=feats, units_store=str(tmp_path / "store"),
                                 expected_counts=wrong, **out)
    out = outputs(tmp_path / "report", feats)
    report_path = str(tmp_path / "counts_vs_expected.json")
    V.prepare_blankfree_data(ogg_zips=zips, feature_hdfs=feats, units_store=str(tmp_path / "store"),
                             expected_counts=wrong, counts_report_path=report_path, **out)
    for split in feats:  # the job ran to the end: every output HDF was written
        assert os.path.exists(out["out_feature_hdfs"][split][0])
    rep = json.load(open(report_path))
    summary = json.load(open(out["manifest_path"]))["summary"]
    assert rep["all_equal"] is False
    assert rep["splits"]["train"]["utterances"] == {"observed": 2, "expected": 3, "rel_diff": -1 / 3}
    kept = summary["train"]["kept_frames"]
    assert rep["splits"]["train"]["kept_frames"] == {"observed": kept, "expected": 1, "rel_diff": kept - 1.0}
    assert rep["splits"]["dev-other"]["utterances"]["rel_diff"] == 0.0


def _clear_input_getters():
    from i6_experiments.users.wu.experiments.unsupervised_asr import inputs as I
    from i6_experiments.users.wu.experiments.unsupervised_asr.data import librispeech as L

    for f in (I.get_inputs, I.get_seed_inputs, I.get_graph, L.get_checked_ffmpeg_binary, L.get_bliss_corpus,
              L.get_ogg_zip):
        f.cache_clear()


@pytest.fixture
def fresh_inputs(monkeypatch, private_jobs):
    monkeypatch.setattr(gs, "FFMPEG_BINARY", "/pinned/by/the/test/ffmpeg", raising=False)
    _clear_input_getters()
    yield
    _clear_input_getters()


def test_inputs_vad_is_report_only_exactly_under_the_label(monkeypatch, fresh_inputs):
    from i6_experiments.users.wu.experiments.unsupervised_asr import inputs as I

    monkeypatch.delattr(gs, "FFMPEG_PIN_ACCEPT", raising=False)
    strict = I.get_inputs().vad
    assert strict.counts_report_only is False and strict.out_counts_report is None
    assert strict.expected_counts == V.BANKED_VAD_COUNTS

    monkeypatch.setattr(gs, "FFMPEG_PIN_ACCEPT", "x86_64-ffmpeg-7.1.1", raising=False)
    _clear_input_getters()
    labelled = I.get_inputs().vad
    assert labelled.counts_report_only is True and labelled.out_counts_report is not None
    assert labelled.expected_counts == V.BANKED_VAD_COUNTS
    assert labelled._sis_id() != strict._sis_id()


# ================================================================================================
# p0's dev-other PER read
# ================================================================================================
@pytest.fixture
def registered(monkeypatch, fresh_inputs):
    out = {}

    def _reg(name, value, export_graph=False):
        assert name not in out or out[name] is value, f"output {name} registered with two values"
        out[name] = value

    monkeypatch.setattr(tk, "register_output", _reg)
    monkeypatch.setattr(tk, "register_report", lambda *a, **k: None)
    return out


def test_p0_per_read_is_the_arms_chain_on_the_selected_export(registered):
    from i6_core.returnn.training import GetBestPtCheckpointJob

    from i6_experiments.users.wu.experiments.unsupervised_asr.analysis.per import BlankfreeGreedyPerJob
    from i6_experiments.users.wu.experiments.unsupervised_asr.config import base, supervised_init
    from i6_experiments.users.wu.experiments.unsupervised_asr.config.common import train_and_read
    from i6_experiments.users.wu.experiments.unsupervised_asr.inputs import get_inputs

    res = supervised_init.py()
    rec, per = res["p0"], res["p0_per"]
    prefix = "sae/4a/analysis_only/p0"
    assert registered[f"{prefix}/best/dev-other/per.json"] is per["per"].out_per
    assert registered[f"{prefix}/selected_epoch"] is rec["best"].out_epoch
    assert isinstance(rec["best"], GetBestPtCheckpointJob)

    # the posterior dump reads the SELECTED, exported recognizer
    post = per["post"]
    assert rec["checkpoint"] is rec["export"].out_checkpoint
    assert post.model_checkpoint.path == rec["checkpoint"]
    assert post.model_checkpoint.path.creator is rec["export"]
    assert rec["export"].checkpoint == rec["best"].out_checkpoint.path

    # the same chain as an arm's read: equal forward config (so equal RETURNN, NET_ARGS, stream and
    # batching), equal PER-job inputs apart from the posteriors
    inputs = get_inputs()
    base.register_inputs(inputs)
    arm = train_and_read(arms.ctrl_20(data=inputs.data), inputs, gaps=False)["per"][20]
    assert sis_hash_helper(post.returnn_config) == sis_hash_helper(arm["post"].returnn_config)
    assert post.returnn_python_exe == arm["post"].returnn_python_exe
    assert post.returnn_root == arm["post"].returnn_root
    assert post.rqmt == arm["post"].rqmt
    p, a = per["per"], arm["per"]
    assert isinstance(p, BlankfreeGreedyPerJob)
    assert (p.features, p.originals, p.gold, p.split) == (a.features, a.originals, a.gold, a.split)
    assert p.posteriors == post.out_files["posteriors.hdf"]
