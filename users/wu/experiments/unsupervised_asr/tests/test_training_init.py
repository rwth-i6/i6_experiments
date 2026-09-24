"""theta's flat init (test plan 2026-09-24, T2.12): ``training/init.py`` ``FlatRecognizerInitJob``.

The job is run in-process with its outputs redirected into tmp_path.  Its checkpoint must load
strictly into the recognizer it was built for and give ``log_q = -log 40`` at every frame in train
AND eval mode (dropout and batch norm active or not: the logit layer is zero), and the bed's own
checkpoint (``config.NET_ARGS``) must load through the model's strict ``recognizer_checkpoint_path``.
"""

from __future__ import annotations

import math
import os

import numpy as np
import pytest
import torch
from sisyphus import tk

from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior
from i6_experiments.users.wu.experiments.unsupervised_asr.model.recognizer import ConvRecognizer
from i6_experiments.users.wu.experiments.unsupervised_asr.training.config import NET_ARGS
from i6_experiments.users.wu.experiments.unsupervised_asr.training.init import FlatRecognizerInitJob

SMALL = dict(NET_ARGS, in_dim=16)


def _run(tmp, net_args, seed=0):
    job = FlatRecognizerInitJob(net_args=net_args, seed=seed)
    job.out_checkpoint = tk.Path(os.path.join(tmp, f"flat_{net_args['in_dim']}_{seed}.pt"))
    job.out_stats = tk.Path(os.path.join(tmp, f"flat_{net_args['in_dim']}_{seed}.stats.txt"))
    job.run()
    return job, torch.load(job.out_checkpoint.get_path(), map_location="cpu", weights_only=False)


def _uniform(shape):
    return torch.full(shape, -math.log(40), dtype=torch.float32)


def test_t2_12_flat_init_small(tmp_path):
    job, ckpt = _run(str(tmp_path), SMALL)
    assert (ckpt["epoch"], ckpt["step"]) == (0, 0) and set(ckpt) >= {"model", "epoch", "step"}
    model = ConvRecognizer(**SMALL)
    model.load_state_dict(ckpt["model"], strict=True)
    assert float(model.conv.weight.abs().max()) == 0.0
    x = torch.randn(3, 20, 16, generator=torch.Generator().manual_seed(0))
    lens = torch.tensor([20, 9, 1])
    for mode in ("train", "eval"):
        getattr(model, mode)()
        with torch.no_grad():
            log_q = model(x, lens)
        assert log_q.shape == (3, 7, 40)
        assert torch.allclose(log_q, _uniform(log_q.shape), rtol=0, atol=1e-6), mode
        assert float(log_q.max() - log_q.min()) == 0.0, mode
    assert "uniform log-prob = -log 40" in open(job.out_stats.get_path()).read()


def test_t2_12_bed_flat_init_loads_through_the_model(tmp_path):
    tmp = str(tmp_path)
    _, ckpt = _run(tmp, NET_ARGS)
    rng = np.random.RandomState(0)
    prior = PhoneNgramPrior(np.log(np.ones(40) / 40), np.log(rng.dirichlet(np.ones(40), size=41)),
                            np.log(rng.dirichlet(np.ones(40), size=41 * 41)), meta={"test": True})
    prior.save(os.path.join(tmp, "prior.npz"))
    np.savez_compressed(os.path.join(tmp, "eta.npz"), tags=np.array(["u0"]), eta=np.zeros((1, 4), np.float32))
    model = BM.SaeBlankfreeModelV1(
        temperature_schedule=[8.0], anchor_weight_schedule=0.0, lam_agg=0.1, count_ema_decay=0.99,
        prior_npz_path=os.path.join(tmp, "prior.npz"), eta_table_path=os.path.join(tmp, "eta.npz"),
        reverse_kwargs={"n_units": 20, "eta_dim": 4, "d_model": 16, "d_ff": 16},
        recognizer_checkpoint_path=os.path.join(tmp, "flat_1024_0.pt"),
        lattice_reduction="matmul", lattice_checkpoint=2)
    for k, v in ckpt["model"].items():
        assert torch.equal(model.recognizer.state_dict()[k], v), k
    x = torch.randn(2, 8, 1024, generator=torch.Generator().manual_seed(1)).half()
    lens = torch.tensor([8, 4])
    for mode in ("train", "eval"):
        getattr(model.recognizer, mode)()
        with torch.no_grad():
            log_q = model.recognizer(x, lens)
        assert torch.allclose(log_q, _uniform(log_q.shape), rtol=0, atol=1e-6), mode

