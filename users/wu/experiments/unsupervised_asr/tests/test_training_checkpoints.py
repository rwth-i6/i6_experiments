"""Checkpoint slicing (test plan 2026-09-24, T2.5): ``training/checkpoints.py``
(``ExtractSubmoduleCheckpointJob``, ``theta_checkpoint``) and ``genmarg_steps.load_phi_state``.

A whole blank-free model (``SaeBlankfreeModelV1``, tiny phi) is saved in RETURNN's checkpoint layout
with NON-default batch-norm statistics and agg EMA buffers, then sliced with the job itself (run
in-process, outputs redirected into tmp_path).  The slices must hold exactly the submodule's
``state_dict`` (BN statistics included, ``agg.*`` excluded), load strictly and reproduce the
submodule's eval outputs bitwise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
from i6_experiments.users.wu.experiments.unsupervised_asr.model.lattice import build_segment_table
from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior
from i6_experiments.users.wu.experiments.unsupervised_asr.model.recognizer import ConvRecognizer
from i6_experiments.users.wu.experiments.unsupervised_asr.model.recognizer_only import get_model as recognizer_get_model
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import N_TYPES
from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg_steps as GS
from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms, jobs
from i6_experiments.users.wu.experiments.unsupervised_asr.training.checkpoints import (
    ExtractSubmoduleCheckpointJob, theta_checkpoint)
from i6_experiments.users.wu.experiments.unsupervised_asr.training.config import NET_ARGS

TAGS, ETA_DIM, N_UNITS = ["u0", "u1"], 4, 20
REVERSE = {"n_units": N_UNITS, "eta_dim": ETA_DIM, "d_model": 16, "d_ff": 16}
EPOCH, STEP = 7, 1234


def _model_kwargs(tmp):
    rng = np.random.RandomState(0)
    n = N_TYPES
    prior = PhoneNgramPrior(
        np.log(np.ones(n) / n), np.log(rng.dirichlet(np.ones(n), size=n + 1)),
        np.log(rng.dirichlet(np.ones(n), size=(n + 1) * (n + 1))), meta={"test": True})
    prior_path, eta_path = os.path.join(tmp, "prior.npz"), os.path.join(tmp, "eta.npz")
    prior.save(prior_path)
    np.savez_compressed(eta_path, tags=np.array(TAGS), eta=rng.randn(2, ETA_DIM).astype(np.float32))
    return dict(temperature_schedule=[8.0, 2.0], anchor_weight_schedule=0.0, lam_agg=0.1,
                count_ema_decay=0.99, band=25, prior_weight=1.0, prior_npz_path=prior_path,
                eta_table_path=eta_path, reverse_kwargs=dict(REVERSE), lam_rate=3.0,
                rate_rho_hz=9.6619373279, lattice_reduction="matmul", lattice_checkpoint=32,
                lattice_float64=True)


@pytest.fixture(scope="module")
def bed(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("ckpt"))
    kw = _model_kwargs(tmp)
    torch.manual_seed(0)
    model = BM.SaeBlankfreeModelV1(**kw)
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():  # non-default buffers everywhere a slice could lose or mix them up
        bn = model.recognizer.bn
        bn.running_mean.copy_(torch.randn(bn.running_mean.shape, generator=g))
        bn.running_var.copy_(torch.rand(bn.running_var.shape, generator=g) + 0.5)
        bn.num_batches_tracked.fill_(17)
        for p in model.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=g))
        model.agg.ema_uni.copy_(torch.rand(model.agg.ema_uni.shape, generator=g))
        model.agg.ema_steps.fill_(5)
    whole = os.path.join(tmp, "epoch.007.pt")
    torch.save({"model": model.state_dict(), "epoch": EPOCH, "step": STEP, "effective_learning_rate": 1e-4},
               whole)
    return dict(tmp=tmp, kw=kw, model=model, whole=whole)


def _extract(bed, prefix, name, **kw):
    job = ExtractSubmoduleCheckpointJob(checkpoint=tk.Path(bed["whole"]), prefix=prefix, **kw)
    job.out_checkpoint = tk.Path(os.path.join(bed["tmp"], f"{name}.pt"))
    job.out_stats = tk.Path(os.path.join(bed["tmp"], f"{name}.stats.txt"))
    job.run()
    return job, torch.load(job.out_checkpoint.get_path(), map_location="cpu", weights_only=False)


def test_whole_checkpoint_layout(bed):
    sd = torch.load(bed["whole"], map_location="cpu", weights_only=False)["model"]
    assert {k.split(".")[0] for k in sd} == {"recognizer", "reverse", "agg"}
    assert "prior_log_bi" not in sd and "eta_table" not in sd  # non-persistent inputs


def test_recognizer_slice_keys_values_and_strict_eval(bed):
    model = bed["model"]
    _, out = _extract(bed, "recognizer.", "theta")
    assert set(out) == {"model", "epoch", "step"} and (out["epoch"], out["step"]) == (EPOCH, STEP)
    ref = model.recognizer.state_dict()
    assert set(out["model"]) == set(ref)
    assert {"bn.running_mean", "bn.running_var", "bn.num_batches_tracked"} <= set(out["model"])
    assert not any(k.startswith("agg") or "ema" in k for k in out["model"])
    for k, v in ref.items():
        assert torch.equal(out["model"][k], v), k
    x = torch.randn(3, 11, 1024, generator=torch.Generator().manual_seed(2))
    lens = torch.tensor([11, 7, 1])
    with torch.no_grad():
        expected = model.recognizer.eval()(x, lens)
    for fresh in (ConvRecognizer(**NET_ARGS), recognizer_get_model(epoch=EPOCH, step=STEP, **NET_ARGS)):
        fresh.load_state_dict(out["model"], strict=True)
        with torch.no_grad():
            got = fresh.eval()(x, lens)
        assert torch.equal(got, expected)
    model.train()


def test_reverse_slice_loads_through_reverse_checkpoint_path(bed):
    model = bed["model"]
    job, out = _extract(bed, "reverse.", "phi")
    assert (out["epoch"], out["step"]) == (EPOCH, STEP)
    ref = model.reverse.state_dict()
    assert set(out["model"]) == set(ref)
    stats = open(job.out_stats.get_path()).read()
    assert f"kept keys  = {len(ref)} of " in stats and "epoch = 7  step = 1234" in stats
    torch.manual_seed(99)  # a different init, so a silent miss would show
    loaded = BM.SaeBlankfreeModelV1(**bed["kw"], reverse_checkpoint_path=job.out_checkpoint.get_path())
    for k, v in ref.items():
        assert torch.equal(loaded.reverse.state_dict()[k], v), k
    units = torch.randint(0, N_UNITS, (2, 9), generator=torch.Generator().manual_seed(3))
    eta = model.lookup_eta(TAGS, "cpu")
    with torch.no_grad():
        assert torch.equal(build_segment_table(loaded.reverse, units, eta),
                           build_segment_table(model.reverse, units, eta))


def test_expect_min_keys_fires(bed):
    n = len(bed["model"].recognizer.state_dict())
    with pytest.raises(AssertionError, match="0 key"):
        _extract(bed, "recogniser.", "typo")
    with pytest.raises(AssertionError, match=f"{n} key"):
        _extract(bed, "recognizer.", "too_many", expect_min_keys=n + 1)
    _extract(bed, "recognizer.", "exact", expect_min_keys=n)


def test_theta_checkpoint_refuses_unkept_epoch():
    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    data = dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"))
    train = jobs.train_arm(**arms.ctrl_20(data=data).job_kwargs(), alias_prefix=None)
    kept = sorted(train.out_checkpoints)
    assert 2 not in kept and 4 in kept
    with pytest.raises(AssertionError, match="not a kept checkpoint"):
        theta_checkpoint(train, 2)
    job, pt = theta_checkpoint(train, 4)
    assert isinstance(job, ExtractSubmoduleCheckpointJob) and job.prefix == "recognizer."
    assert job.checkpoint == train.out_checkpoints[4].path
    assert pt.path == job.out_checkpoint


def test_load_phi_state_same_from_sliced_and_whole(bed):
    job, _ = _extract(bed, "reverse.", "phi2")
    sliced, meta_s = GS.load_phi_state(job.out_checkpoint.get_path())
    whole, meta_w = GS.load_phi_state(bed["whole"])
    assert (meta_s["format"], meta_w["format"]) == ("reverse_only", "whole_model")
    assert (meta_s["epoch"], meta_s["step"]) == (meta_w["epoch"], meta_w["step"]) == (EPOCH, STEP)
    assert set(sliced) == set(whole) == set(bed["model"].reverse.state_dict())
    for k in whole:
        assert torch.equal(sliced[k], whole[k]), k
