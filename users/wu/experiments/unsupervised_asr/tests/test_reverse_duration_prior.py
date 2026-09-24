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


# ===================================================================================================
# Priority-2 test (test plan 2026-09-24, T2.13): the model option reverse_duration_prior
# (model/blankfree_model.py) -- init law, freeze, refusals.
# ===================================================================================================

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

import torch  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior  # noqa: E402

RHO_HZ = 9.6619373279


@pytest.fixture(scope="module")
def dur_bed(tmp_path_factory):
    tmp = str(tmp_path_factory.mktemp("durprior"))
    rng = np.random.RandomState(0)
    prior = PhoneNgramPrior(np.log(np.ones(40) / 40), np.log(rng.dirichlet(np.ones(40), size=41)),
                            np.log(rng.dirichlet(np.ones(40), size=41 * 41)), meta={"test": True})
    prior.save(os.path.join(tmp, "prior.npz"))
    np.savez_compressed(os.path.join(tmp, "eta.npz"), tags=np.array(["u0"]), eta=np.zeros((1, 4), np.float32))
    mean = D.prior_mean_frames(retained_frames=80, original_frames=100, rho_hz=RHO_HZ, frame_rate_hz=50.0)
    json.dump({"mean_frames": mean}, open(os.path.join(tmp, "prior.json"), "w"))
    kw = dict(temperature_schedule=[8.0], anchor_weight_schedule=0.0, lam_agg=0.1, count_ema_decay=0.99,
              prior_npz_path=os.path.join(tmp, "prior.npz"), eta_table_path=os.path.join(tmp, "eta.npz"),
              reverse_kwargs={"n_units": 20, "eta_dim": 4, "d_model": 16, "d_ff": 16},
              lattice_reduction="matmul", lattice_checkpoint=2)
    return dict(kw=kw, json=os.path.join(tmp, "prior.json"), mean=mean)


def _dur_model(dur_bed, mode, **over):
    torch.manual_seed(0)
    return BM.SaeBlankfreeModelV1(**dur_bed["kw"], reverse_duration_prior=dur_bed["json"],
                                  reverse_duration_prior_mode=mode, **over)


def _oracle_law(d_min, d_max, mean):
    """p(d) ∝ exp(lambda d) on [d_min, d_max] with mean ``mean``: lambda by scipy's brentq, float64."""
    from scipy.optimize import brentq

    d = np.arange(d_min, d_max + 1, dtype=np.float64)

    def law(lam):
        w = np.exp(lam * d - (lam * d).max())
        return w / w.sum()

    lam = brentq(lambda x: float((law(x) * d).sum()) - mean, -50.0, 50.0, xtol=1e-15, rtol=1e-15, maxiter=500)
    return law(lam)


def test_t2_13_initial_law(dur_bed):
    m = dur_bed["mean"]
    assert abs(m - (50.0 / RHO_HZ) * 0.8) < 1e-12
    model = _dur_model(dur_bed, "init")
    cfg = model.reverse.cfg
    sil = cfg.sil_id
    table = BM.max_entropy_duration_logits(cfg, m)  # float64, the law the constructor writes
    oracle = _oracle_law(cfg.d_min, cfg.d_max, m)
    lp = model.reverse.duration_log_probs().detach().double()
    worst64 = worst32 = 0.0
    for k in range(cfg.n_types):
        if k == sil:
            continue
        legal = slice(cfg.d_min - 1, cfg.d_max)
        p64 = torch.softmax(table[k, legal], 0).numpy()
        worst64 = max(worst64, float(np.abs(p64 - oracle).max()))
        assert np.abs(p64 - oracle).max() <= 1e-9, k
        assert torch.equal(model.reverse.dur_logits.detach()[k], table[k].to(torch.float32)), k
        p32 = lp[k].exp().numpy()
        worst32 = max(worst32, float(np.abs(p32[legal] - oracle).max()))
        assert np.abs(p32[legal] - oracle).max() <= 1e-6, k
        assert p32[: cfg.d_min - 1].sum() == 0.0 and p32[cfg.d_max:].sum() == 0.0
    print(f"T2.13 phone law vs brentq oracle: float64 table {worst64:.2e}, model float32 {worst32:.2e}")
    p_sil = lp[sil].exp().numpy()
    assert np.abs(p_sil[cfg.d_min - 1: cfg.d_max_sil] - 1.0 / (cfg.d_max_sil - cfg.d_min + 1)).max() <= 1e-7
    assert p_sil[: cfg.d_min - 1].sum() == 0.0 and (cfg.d_min, cfg.d_max_sil) == (2, 50)


def _one_adam_step(model):
    """One step of the bed's optimizer settings (Adam, betas (0.5, 0.98), eps 1e-6, weight decay 0,
    phi's lr 3e-3) on a loss through every legal duration cell.  torch's Adam, not RETURNN's updater."""
    rev = model.reverse
    legal = rev.duration_log_probs().detach() > -1e10
    w = torch.rand(legal.shape, generator=torch.Generator().manual_seed(5))
    opt = torch.optim.Adam([p for p in rev.parameters() if p.requires_grad], lr=3e-3, betas=(0.5, 0.98),
                           eps=1e-6, weight_decay=0.0)
    before = rev.dur_logits.detach().clone()
    loss = -(w * torch.where(legal, rev.duration_log_probs(), torch.zeros(()))).sum()
    opt.zero_grad()
    loss.backward()
    grad = rev.dur_logits.grad.clone()
    opt.step()
    return before, rev.dur_logits.detach().clone(), grad


def test_t2_13_freeze_and_init_after_one_adam_step(dur_bed):
    frz = _dur_model(dur_bed, "freeze")
    sil = frz.reverse.cfg.sil_id
    phone = torch.arange(frz.reverse.cfg.n_types) != sil
    before, after, grad = _one_adam_step(frz)
    assert float(grad[phone].abs().max()) == 0.0 and float(grad[sil].abs().max()) > 0.0
    assert torch.equal(before[phone], after[phone])  # bitwise unchanged
    assert not torch.equal(before[sil], after[sil])
    ini = _dur_model(dur_bed, "init")
    before, after, grad = _one_adam_step(ini)
    assert float(grad[phone].abs().max()) > 0.0
    assert not torch.equal(before[phone], after[phone]) and not torch.equal(before[sil], after[sil])


@pytest.mark.parametrize("over, match", [
    ({"reverse_checkpoint_path": "/nonexistent/phi.pt"}, "refused with a loaded phi"),
    ({"freeze_reverse": True}, "freeze_reverse"),
])
def test_t2_13_refusals(dur_bed, over, match):
    for mode in ("init", "freeze"):
        with pytest.raises(AssertionError, match=match):
            _dur_model(dur_bed, mode, **over)
