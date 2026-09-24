"""Tests of ``training/config.py``, ``arms.py`` and ``jobs.py`` at graph-build level (no job runs):
the get_model keyword order the banked configs were written in, the removed BT flags, the seed
knobs, the preset deltas and the job's interpreter / resources."""

import pytest
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr import default_tools
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
        train_segments=tk.Path("/nonexistent/train.segments"),
        dev_segments=tk.Path("/nonexistent/cv.segments"),
        prior_npz=tk.Path("/nonexistent/prior.npz"),
        eta_npz=tk.Path("/nonexistent/eta.npz"),
        flat_checkpoint=tk.Path("/nonexistent/flat_init.pt"),
    )


def _graph(theta):
    return dict(
        hlg=tk.Path("/nonexistent/HLG.pt"),
        stats=tk.Path("/nonexistent/build.json"),
        resources=tk.Path("/nonexistent/lexlat_resources.npz"),
        expected_build={"backoff_loops": "word_boundary", "escape": True, "sil_prob": 0.5, "theta": theta},
    )


BASE_KEYS = [
    "temperature_schedule", "anchor_weight_schedule", "lam_agg", "count_ema_decay", "band",
    "prior_weight", "prior_npz_path", "eta_table_path", "recognizer_checkpoint_path", "lam_rate",
    "rate_rho_hz", "rate_fd_eps", "rate_fd_mode", "lattice_reduction", "lattice_checkpoint",
    "lattice_float64",
]
K2_KEYS = [
    "lexlat_k2_hlg", "lexlat_k2_stats", "lexlat_k2_resources", "lexlat_k2_expected_build",
    "lexlat_k2_max_active", "lexlat_k2_onset", "lexlat_k2_ramp", "lexlat_k2_full_lam",
    "lexlat_k2_search_beam", "lexlat_k2_output_beam", "lexlat_k2_min_active_states",
]


def test_bed_defaults():
    cfg = config.build_train_config(**_data())
    args = jobs.get_model_args(cfg)
    assert list(args) == BASE_KEYS
    assert args["temperature_schedule"] == config.DEFAULT_TEMPERATURE_SCHEDULE
    assert cfg.config["learning_rates"] == [1e-4] * 4
    assert cfg.config["optimizer"]["betas"] == [0.5, 0.98] and cfg.config["optimizer"]["eps"] == 1e-6
    assert "random_seed" not in cfg.config
    with pytest.raises(AssertionError):
        config.build_train_config(**_data(), num_subepochs=20)  # default tau is 4 entries


@pytest.mark.parametrize(
    "kw", [dict(lam_bt=0.5), dict(bt_depth="output_only"), dict(bt_ramp_epochs=2), dict(bt_batch_sents=64),
           dict(bt_text_phn=tk.Path("/nonexistent/text"))]
)
def test_removed_bt_flags_raise(kw):
    with pytest.raises(ValueError):
        config.build_train_config(**_data(), **kw)


def test_stage1_phi_first_deltas():
    """The L2-1 stage-1 durinit restart as banked (PhiFirstProbeTrainingJob.zWqS49iSFTdV)."""
    cfg = config.build_train_config(
        **_data(),
        num_subepochs=4,
        temperature_schedule=[4.0, 1.0, 1.0, 1.0],
        null_recognizer=True,
        freeze_recognizer=True,
        reverse_duration_prior=tk.Path("/nonexistent/prior.json"),
        reverse_duration_prior_mode="init",
        adam_betas=(0.9, 0.999),
        adam_eps=1e-8,
        random_seed=1,
        random_seed_offset=1,
    )
    args = jobs.get_model_args(cfg)
    assert list(args) == BASE_KEYS + [
        "null_recognizer", "freeze_recognizer", "reverse_duration_prior", "reverse_duration_prior_mode"
    ]
    c = cfg.config
    assert c["optimizer"]["betas"] == [0.9, 0.999] and c["optimizer"]["eps"] == 1e-08
    assert c["random_seed"] == 1
    feats = c["train"]["dataset"]["datasets"]["feats"]
    assert feats["random_seed_offset"] == 1
    assert "random_seed_offset" not in c["dev"]["datasets"]["feats"]
    with pytest.raises(AssertionError):
        config.build_train_config(**_data(), reverse_duration_prior_mode="init")


def test_ladder_order_reverse_ckpt_before_k2_chunk_last():
    k2 = config.lexlat_k2_model_args(**_graph(0.0), max_active=1000, onset=1, chunk_seqs=4)
    cfg = config.build_train_config(
        **_data(), reverse_checkpoint_path=tk.Path("/nonexistent/epoch.008.pt"), lexlat_k2=k2
    )
    assert list(jobs.get_model_args(cfg)) == BASE_KEYS + ["reverse_checkpoint_path"] + K2_KEYS + [
        "lexlat_k2_chunk_seqs"
    ]


def test_presets():
    ctrl = arms.ctrl_20(data=_data())
    assert (ctrl.name, ctrl.num_epochs, ctrl.keep_epochs) == ("ctrl_20", 20, (1, 4, 10, 20))
    assert list(jobs.get_model_args(ctrl.returnn_config)) == BASE_KEYS

    k2 = arms.k2lat_20_ma3000(data=_data(), graph=_graph(0.0))
    args = jobs.get_model_args(k2.returnn_config)
    assert list(args) == BASE_KEYS + K2_KEYS  # "full": no prior_weight_schedule key
    assert (args["lexlat_k2_max_active"], args["lexlat_k2_onset"]) == (3000, 8)

    off4 = arms.off4_k2lat_20(data=_data(), graph=_graph(5.0))
    assert jobs.get_model_args(off4.returnn_config)["lexlat_k2_max_active"] == 1000
    with pytest.raises(AssertionError):
        arms.off4_k2lat_20(data=_data(), graph=_graph(0.0))  # the in-house graph is not the official one

    x60 = arms.k2lat_20_ma3000(data=_data(), graph=_graph(0.0), num_sub_epochs=60)
    assert (x60.name, x60.num_epochs, x60.keep_epochs) == ("k2lat_20_ma3000_x60", 60, (1, 4, 10, 20, 30, 40, 50, 60))
    assert len(x60.returnn_config.config["learning_rates"]) == 60
    with pytest.raises(ValueError):
        arms.ctrl_20(data=_data(), num_sub_epochs=40)


def test_word_lm_modes():
    for mode, expected in [("full", None), ("off", [0.0] * 20)]:
        arm = arms.k2_word_lm(data=_data(), graph=_graph(5.0), phone_trigram=mode)
        assert jobs.get_model_args(arm.returnn_config).get("prior_weight_schedule") == expected, mode
    with pytest.raises(ValueError):
        arms.k2_word_lm(data=_data(), graph=_graph(5.0), phone_trigram="half")


def test_word_lm_rampout():
    arm = arms.k2_word_lm(data=_data(), graph=_graph(5.0))
    args = jobs.get_model_args(arm.returnn_config)
    assert list(args)[-1] == "prior_weight_schedule"
    assert args["prior_weight_schedule"] == [1.0] * 7 + [0.6666666666666667, 0.33333333333333337] + [0.0] * 11
    assert args["prior_weight"] == 1.0


def test_train_arm_resources_and_exe():
    ctrl = jobs.train_arm(**arms.ctrl_20(data=_data()).job_kwargs(), alias_prefix=None)
    k2 = jobs.train_arm(**arms.k2lat_20_ma3000(data=_data(), graph=_graph(0.0)).job_kwargs(), alias_prefix=None)
    for job, exe in ((ctrl, default_tools.SAE_PYTHON_EXE), (k2, default_tools.K2_PYTHON_EXE)):
        assert job.returnn_python_exe.get_path() == exe.get_path()
        assert job.returnn_python_exe.hash_overwrite == exe.hash_overwrite
    for job in (ctrl, k2):
        assert job.rqmt["time"] == 11.5 and job.rqmt["mem"] == 64 and job.rqmt["cpu"] == 16
        assert job.rqmt["gpu_mem"] == 96 and job.rqmt["gpu"] == 1
        assert job.returnn_config.post_config["log_verbosity"] == 5
        assert job.returnn_config.post_config["cleanup_old_models"]["keep"] == [1, 4, 10, 20]
        assert job.returnn_root.hash_overwrite == default_tools.RETURNN_ROOT.hash_overwrite
    with pytest.raises(AssertionError):
        jobs.train_arm("bad", arms.ctrl_20(data=_data()).returnn_config, 60, keep_epochs=(1,), alias_prefix=None)
