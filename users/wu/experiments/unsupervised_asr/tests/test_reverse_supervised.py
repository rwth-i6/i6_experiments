"""Tests of the ANALYSIS ONLY (uses transcripts) inits at graph-build level: the supervised reverse
fit's RETURNN config and the p0 recognizer's config deltas (no job runs)."""

from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import p0, supervised
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model.supervised_data import with_edge_sil
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model.supervised_steps import FIT
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import SIL_ID


def test_fit_constants():
    assert (FIT.seed, FIT.epochs, FIT.lr, FIT.grad_clip, FIT.weight_decay) == (42, 8, 3e-3, 5.0, 0.0)
    assert with_edge_sil([1, 2]) == [SIL_ID, 1, 2, SIL_ID]


def test_supervised_reverse_config():
    cfg = supervised.supervised_reverse_config(train_hdf=tk.Path("/nonexistent/train.hdf"),
                                               held_hdf=tk.Path("/nonexistent/held.hdf"))
    c = cfg.config
    assert c["learning_rates"] == [3e-3] * 8 and c["gradient_clip_global_norm"] == 5.0
    assert c["optimizer"] == {"class": "adam", "weight_decay": 0.0}
    assert c["train"]["datasets"]["seed"]["partition_epoch"] == 8
    assert c["dev"]["datasets"]["seed"]["partition_epoch"] == 1


def test_blankfree_supervised_reverse_init_jobs():
    p = lambda n: tk.Path(f"/nonexistent/{n}")  # noqa: E731
    data, train = supervised.blankfree_supervised_reverse_init(
        gold_json=p("gold.json"), ids_json=p("ids.json"), targets_hdf=p("targets.hdf"),
        train_segments=p("train.segments"), cv_segments=p("cv.segments"), units_hdfs=[p("u.hdf")],
        eta_npz=p("eta.npz"), alias=None)
    assert sorted(train.out_checkpoints) == list(range(1, 9))
    assert train.rqmt["gpu_mem"] == supervised.GPU_MEM_RQMT


def _data():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


def test_p0_config_deltas():
    cfg = p0.p0_train_config(data=_data(), targets_hdf=tk.Path("/nonexistent/targets.hdf"),
                             seed_train_segments=tk.Path("/nonexistent/s.train"),
                             seed_held_segments=tk.Path("/nonexistent/s.held"))
    c = cfg.config
    assert len(c["learning_rates"]) == p0.NUM_SUBEPOCHS == 30
    assert c["newbob_multi_num_epochs"] == 1
    assert c["train"]["seq_order_control_dataset"] == "targets"
    assert c["train"]["datasets"]["targets"]["partition_epoch"] == 1
    assert cfg.post_config["cleanup_old_models"]["keep_best_n"] == p0.KEEP_BEST_N
    out = p0.p0_recognizer(data=_data(), gold_json=tk.Path("/nonexistent/g.json"),
                           ids_json=tk.Path("/nonexistent/i.json"), old_targets_hdf=tk.Path("/nonexistent/t.hdf"),
                           seed_train_segments=tk.Path("/nonexistent/s.train"),
                           seed_held_segments=tk.Path("/nonexistent/s.held"),
                           train_units_hdfs=[tk.Path("/nonexistent/u.hdf")], alias=None)
    assert out["train"].rqmt["time"] == p0.TIME_RQMT and out["export"].prefix == p0.RECOGNIZER_PREFIX
