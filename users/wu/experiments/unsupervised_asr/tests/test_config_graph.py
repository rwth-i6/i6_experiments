"""Graph-build tests of the ``config/`` entry points: each ``py()`` builds its sisyphus graph (CPU, no job
runs; ``tk.register_output`` / ``tk.register_report`` patched), registers the expected outputs, keeps
every arm a job of its own, and the default k2 arm resolves to the official 4-gram graph."""

import ast
import os

import pytest
from sisyphus import gs, tk

from i6_experiments.users.wu.experiments.unsupervised_asr.config import base, common, k2_word_lm, lexlat_v2, supervised_init

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def registered(monkeypatch):
    # a private job registry: the graphs built here must not merge into later tests' jobs of the
    # same hash (e.g. an encode job whose ffmpeg path is hash-invisible)
    import sisyphus.job

    monkeypatch.setattr(sisyphus.job, "created_jobs", {})
    # FFMPEG_BINARY is required by the graph (default_tools.get_ffmpeg_binary); a dummy path, since
    # nothing runs here and the path is hash-invisible
    monkeypatch.setattr(gs, "FFMPEG_BINARY", "/pinned/by/the/test/ffmpeg", raising=False)
    out = {}

    def _reg(name, value, export_graph=False):
        assert name not in out or out[name] is value, f"output {name} registered with two values"
        out[name] = value

    monkeypatch.setattr(tk, "register_output", _reg)
    monkeypatch.setattr(tk, "register_report", lambda *a, **k: None)
    return out


def _model_args(arm):
    from i6_experiments.users.wu.experiments.unsupervised_asr.training.jobs import get_model_args

    return get_model_args(arm.returnn_config)


def test_base(registered):
    res = base.py()
    assert set(res["arms"]) == {"ctrl_20", "ctrl_20_x60"}
    for name, epochs in (("ctrl_20", (1, 4, 10, 20)), ("ctrl_20_x60", (1, 4, 10, 20, 30, 40, 50, 60))):
        for ep in epochs:
            assert f"sae/4a/{name}/ep{ep}/dev-other/per.json" in registered
        final = epochs[-1]
        for f in ("derangement_gap.json", "decode_gap.json", "decode_gap.txt"):
            assert f"sae/4a/{name}/ep{final}/dev-other/{f}" in registered
        assert f"sae/4a/{name}/ep{epochs[-2]}/dev-other/decode_gap.json" not in registered
        assert max(res["arms"][name]["train"].out_checkpoints) == final
    assert "sae/4a/js_rows/ctrl/js_rows.json" in registered
    for key in ("sae/4a/data/vad/manifest.json", "sae/4a/lm/prior.npz", "sae/4a/lexlat_durprior/prior.json"):
        assert key in registered
    # the read-side streams are dev-other's, the training dev set is the train stream under cv
    inputs = res["inputs"]
    assert inputs.data["dev_feature_hdfs"] == inputs.data["train_feature_hdfs"]
    assert inputs.data["dev_segments"] is inputs.cv_split.out_cv_segments
    assert len(inputs.data["train_feature_hdfs"]) == 4


def test_k2_word_lm_default_is_official_4gram(registered):
    from i6_experiments.users.wu.experiments.unsupervised_asr.inputs import get_graph
    from i6_experiments.users.wu.experiments.unsupervised_asr.lm.hlg import (
        LexlatOfficialHLGBuildJob,
        LexlatOfficialResourcesJob,
    )

    res = k2_word_lm.py()
    assert set(res["arms"]) == {"k2_word_lm", "k2lat_20_ma3000", "k2lat_20_ma3000_x60", "off4_k2lat_20"}
    arm = res["arms"]["k2_word_lm"]["arm"]
    args = _model_args(arm)
    official = get_graph("official_4gram")
    assert args["lexlat_k2_hlg"] is official["hlg"]
    assert isinstance(args["lexlat_k2_hlg"].creator, LexlatOfficialHLGBuildJob)
    assert args["lexlat_k2_resources"] is official["resources"]
    assert isinstance(args["lexlat_k2_resources"].creator, LexlatOfficialResourcesJob)
    assert args["lexlat_k2_expected_build"]["theta"] == 5.0
    assert args["lexlat_k2_max_active"] == 1000
    # rampout: full weight before the on-set, 0 from the end of the ramp on
    sched = args["prior_weight_schedule"]
    assert len(sched) == 20 and sched[0] == 1.0 and sched[-1] == 0.0
    # the banked presets write no schedule (scalar prior_weight 1.0)
    for name in ("k2lat_20_ma3000", "k2lat_20_ma3000_x60", "off4_k2lat_20"):
        assert "prior_weight_schedule" not in _model_args(res["arms"][name]["arm"]), name
    inhouse = _model_args(res["arms"]["k2lat_20_ma3000"]["arm"])
    assert inhouse["lexlat_k2_expected_build"]["theta"] == 0.0 and inhouse["lexlat_k2_max_active"] == 3000
    for name, base_name in (("k2_word_lm", "ctrl_20"), ("k2lat_20_ma3000_x60", "ctrl_20_x60")):
        ep = 60 if name.endswith("x60") else 20
        assert f"sae/4a/paired/{name}_vs_{base_name}/ep{ep}/dev-other/paired_per.json" in registered
    assert "sae/4a/js_rows/k2_word_lm_rampout/js_rows.json" in registered


def test_no_two_arms_collapse(registered):
    jobs = {}
    for name, r in {**base.py()["arms"], **k2_word_lm.py()["arms"]}.items():
        jobs[name] = r["train"]
    probe = lexlat_v2.probe()
    for setting, by_seed in probe["probes"].items():
        for seed, job in by_seed.items():
            jobs[f"probe/{setting}/s{seed}"] = job
    for arm, job in lexlat_v2.ladder()["rt"].items():
        jobs[f"ladder/{arm}"] = job
    # a dummy decision (8 sub-epochs); at 4 sub-epochs see the next test
    wave = lexlat_v2.wave(phi_c_source=tk.Path("/nonexistent/phi_c_source.pt"), duration="durinit", num_subepochs=8)
    for group in ("restarts", "nulls", "references"):
        for arm, job in wave[group].items():
            jobs[f"wave/{arm}"] = job
    ids = {name: job.job_id() for name, job in jobs.items()}
    assert len(set(ids.values())) == len(ids), sorted(ids.items(), key=lambda kv: kv[1])
    assert len(ids) == 6 + 6 + 8 + 24


def test_wave_at_probe_length_is_apart_from_the_probe(registered):
    """At 4 sub-epochs (the probe's length) the wave arms and the probe runs are distinct jobs, as in
    the source (separate trainings), and building ``wave(4)`` BEFORE ``probe()`` in one process works:
    the probe keeps its per-sub-epoch checkpoints (a merged job kept only the wave's last one and the
    probe build failed with ``KeyError: 1``)."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first as pf

    wave = lexlat_v2.wave(phi_c_source=tk.Path("/nonexistent/phi_c_source.pt"), duration="durinit", num_subepochs=4)
    probe = lexlat_v2.probe()["probes"]
    wave_jobs = {arm: job for g in ("restarts", "nulls", "references") for arm, job in wave[g].items()}
    probe_ids = {job.job_id() for by_seed in probe.values() for job in by_seed.values()}
    assert len(wave_jobs) == 24 and len(probe_ids) == 6
    assert not probe_ids & {job.job_id() for job in wave_jobs.values()}
    for by_seed in probe.values():
        for job in by_seed.values():
            assert sorted(job.out_checkpoints) == [1, 2, 3, 4]
            assert pf.WAVE_KEY not in job.returnn_config.config
    for arm, job in wave_jobs.items():
        assert sorted(job.out_checkpoints) == [4] and job.returnn_config.config[pf.WAVE_KEY] == 1, arm
    # the tag is the only difference: without it restart s01 hashes as the probe's durinit seed 1
    from sisyphus.hash import sis_hash_helper

    rc = wave["restarts"]["em_s01"].returnn_config.config
    pc = probe["durinit"][1].returnn_config.config
    assert sis_hash_helper({k: v for k, v in rc.items() if k != pf.WAVE_KEY}) == sis_hash_helper(pc)


def test_lexlat_v2_py_builds_probe_only(registered):
    res = lexlat_v2.py()
    assert set(res) == {"probe"}
    assert set(res["probe"]["probes"]) == {"uniform", "durinit", "durfrz"}
    assert any(k.startswith("sae/4a/lexlat_v2/em/probe/durfrz_s01/ep4/") for k in registered)
    assert not any("/wave/" in k or "/ladder/" in k for k in registered)
    assert not any("analysis_only" in k for k in registered)


def test_wave_refuses_undecided_values(registered):
    from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import phi_first

    assert phi_first.WAVE_DURATION_SETTING is None and phi_first.WAVE_NUM_SUBEPOCHS is None
    with pytest.raises(TypeError):
        lexlat_v2.wave()  # every argument is required
    with pytest.raises(AssertionError):
        lexlat_v2.wave(phi_c_source=tk.Path("/nonexistent/x.pt"), duration=None, num_subepochs=None)


def test_supervised_init(registered):
    res = supervised_init.py()
    assert "sae/4a/analysis_only/supervised_goldphi/learning_rates.dev_loss_nll_per_frame" in registered
    assert "sae/4a/analysis_only/p0/recognizer_init.pt" in registered
    assert max(res["gold_phi"]["train"].out_checkpoints) == 8


def test_js_rows_bootstrap_is_the_registered_convention():
    from i6_experiments.users.wu.experiments.unsupervised_asr.analysis.jsd import JS_ROWS_CONVENTION

    assert "2000" in JS_ROWS_CONVENTION and "seed 0" in JS_ROWS_CONVENTION
    assert (common.JS_ROWS_N_BOOT, common.JS_ROWS_SEED) == (2000, 0)


@pytest.mark.parametrize("rel", ["__init__.py", "config/__init__.py"])
def test_package_inits_are_import_free(rel):
    """The k2 child processes import the package under an interpreter without sisyphus."""
    with open(os.path.join(PKG_DIR, rel)) as fh:
        tree = ast.parse(fh.read())
    assert not [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
