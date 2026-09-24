"""Tests of ``model/blankfree_model.py``, ``model/emc_model.py`` and ``model/train_step.py``
on a tiny CPU bed (a random prior and eta table in tmp_path, a
small phi, a 1024-d feature batch of a few frames): construction, one full train step with its
losses and gradients, and the ``ValueError`` on every flag whose term the port removed.

The numerical equivalence with the speech-llm c49559ce source is not tested here (committed tests
never import it); it was checked once by the port's old-vs-new comparison scripts.
"""

import numpy as np
import pytest
import torch

import returnn.frontend as rf
from returnn.frontend._backend import select_backend_torch
from returnn.tensor import Dim, Tensor, TensorDict

from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
from i6_experiments.users.wu.experiments.unsupervised_asr.model import emc_model as EM
from i6_experiments.users.wu.experiments.unsupervised_asr.model import param_groups as PG
from i6_experiments.users.wu.experiments.unsupervised_asr.model import train_step as TS
from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior
from i6_experiments.users.wu.experiments.unsupervised_asr.phones import N_TYPES

B, ETA_DIM, N_UNITS = 2, 4, 20
LENS = [21, 17]
TAGS = ["utt0", "utt1"]


@pytest.fixture(scope="module", autouse=True)
def _backend():
    select_backend_torch()  # mark_as_loss needs a selected backend outside a RETURNN run


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bed")
    rng = np.random.RandomState(0)
    n = N_TYPES
    prior = PhoneNgramPrior(
        np.log(np.ones(n) / n),
        np.log(rng.dirichlet(np.ones(n), size=n + 1)),
        np.log(rng.dirichlet(np.ones(n), size=(n + 1) * (n + 1))),
        meta={"test": True},
    )
    prior_path = str(tmp / "prior.npz")
    prior.save(prior_path)
    eta_path = str(tmp / "eta.npz")
    np.savez_compressed(eta_path, tags=np.array(TAGS), eta=rng.randn(2, ETA_DIM).astype(np.float32))
    return prior_path, eta_path


def _kwargs(inputs, **over):
    prior_path, eta_path = inputs
    kw = dict(
        temperature_schedule=[8.0, 2.0, 2.0], anchor_weight_schedule=0.0, lam_agg=0.1,
        count_ema_decay=0.99, band=4, prior_weight=1.0, prior_npz_path=prior_path,
        eta_table_path=eta_path,
        reverse_kwargs={"n_units": N_UNITS, "d_max": 4, "d_max_sil": 6, "eta_dim": ETA_DIM,
                        "d_model": 16, "d_ff": 16},
        lam_rate=3.0, rate_rho_hz=9.6619373279, rate_fd_eps=0.25, rate_fd_mode="central",
        lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True,
    )
    kw.update(over)
    return kw


def _batch_dims():
    batch = Dim(B, name="batch")
    tdim = Dim(name="time", dimension=None, dyn_size_ext=Tensor(
        "t", raw_tensor=torch.tensor(LENS, dtype=torch.int32), dims=[batch], dtype="int32"))
    return batch, tdim


def _features(batch, tdim, g):
    feat = Dim(1024, name="feat")
    x = torch.randn(B, max(LENS), 1024, generator=g).half()
    return Tensor("features", dims=[batch, tdim, feat], dtype="float16", feature_dim=feat, raw_tensor=x)


def _extern_data(seed=0):
    g = torch.Generator().manual_seed(seed)
    batch, tdim = _batch_dims()
    sdim = Dim(name="units_time", dimension=None, dyn_size_ext=Tensor(
        "s", raw_tensor=torch.tensor(LENS, dtype=torch.int32), dims=[batch], dtype="int32"))
    data = TensorDict()
    data.data["features"] = _features(batch, tdim, g)
    data.data["units"] = Tensor(
        "units", dims=[batch, sdim], dtype="int32", sparse_dim=Dim(N_UNITS, name="u"),
        raw_tensor=torch.randint(0, N_UNITS, (B, max(LENS)), generator=g, dtype=torch.int32))
    odim = Dim(1, name="o")
    data.data["original_length"] = Tensor(
        "original_length", dims=[batch, odim], dtype="int32",
        raw_tensor=torch.tensor([[LENS[0] + 5], [LENS[1]]], dtype=torch.int32))
    tag = Tensor("seq_tag", dims=[batch], dtype="string")
    tag.raw_tensor = np.array(TAGS)
    data.data["seq_tag"] = tag
    return data


def _run(model, step, data, epoch=1):
    model.train()
    torch.manual_seed(0)
    rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=epoch)
    step(model=model, extern_data=data)
    ctx = rf.get_run_ctx()
    total = ctx.total_loss()
    total = total.raw_tensor if hasattr(total, "raw_tensor") else total
    total.backward()
    return ctx.losses, float(total)


def test_construction(inputs):
    model = BM.get_model(epoch=1, step=0, device="cpu", **_kwargs(inputs))
    assert isinstance(model, BM.SaeBlankfreeModelV1) and isinstance(model, EM.SaeEmcModelV1)
    assert model.recognizer.n_out == 40 and model.recognizer.stride == 3
    for gone in ("ent_schedule", "lexlat", "soft", "sf", "lam_bt", "lam_cons", "lam_content",
                 "q_init", "lam_selfdistill", "rate_hinge", "prior_classes",
                 "zero_reverse_emission", "permute_frames_seed"):
        assert not hasattr(model, gone), gone
    assert getattr(model, "lexlat_k2", None) is None
    groups = PG.emc_param_groups(model=model, recognizer_lr_multiplier=1.0, reverse_lr_multiplier=30.0)
    assert [g["learning_rate_multiplier"] for g in groups] == [1.0, 30.0]


def test_unknown_kwarg_is_a_type_error(inputs):
    with pytest.raises(TypeError, match="unknown model_args"):
        BM.SaeBlankfreeModelV1(**_kwargs(inputs, lam_typo=1.0))


def test_train_step_forward_and_gradients(inputs):
    model = BM.SaeBlankfreeModelV1(**_kwargs(inputs))
    losses, total = _run(model, TS.train_step, _extern_data())
    for key in ("l_tau", "agg", "rate", "blankfree_l_tau_per_frame", "blankfree_rate_fd_check",
                "blankfree_expected_phone_rate_hz", "blankfree_temperature"):
        assert key in losses, key
    assert float(losses["blankfree_temperature"].loss.raw_tensor.sum()) == 8.0
    assert np.isfinite(total)
    for side in (model.recognizer, model.reverse):
        grads = [p.grad for p in side.parameters() if p.requires_grad]
        assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
        assert any(float(g.abs().sum()) > 0 for g in grads)


def test_null_recognizer_step(inputs):
    model = BM.SaeBlankfreeModelV1(**_kwargs(inputs, null_recognizer=True, freeze_recognizer=True))
    losses, total = _run(model, TS.train_step, _extern_data())
    assert "blankfree_nonsil_rate_retained_hz" in losses and "blankfree_rate_fd_check" not in losses
    assert losses["rate"].as_error and np.isfinite(total)
    assert all(p.grad is None for p in model.recognizer.parameters())


def _non_default(value):
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, tuple):
        return ("specaug", "speed") if value else ("specaug",)
    return "set"  # None or a string default


@pytest.mark.parametrize("flag", sorted(BM.REMOVED_BLANKFREE_FLAGS))
def test_removed_blankfree_flag_raises(inputs, flag):
    bad = _non_default(BM.REMOVED_BLANKFREE_FLAGS[flag])
    with pytest.raises(ValueError, match=flag):
        BM.SaeBlankfreeModelV1(**_kwargs(inputs, **{flag: bad}))


@pytest.mark.parametrize("flag", sorted(EM.REMOVED_EMC_FLAGS))
def test_removed_emc_flag_raises(inputs, flag):
    bad = _non_default(EM.REMOVED_EMC_FLAGS[flag])
    with pytest.raises(ValueError, match=flag):
        BM.SaeBlankfreeModelV1(**_kwargs(inputs, **{flag: bad}))


@pytest.mark.parametrize("over, name", [
    ({"anchor_weight_schedule": 0.5}, "anchor_weight_schedule"),
    ({"anchor_weight_schedule": [0.0, 1.0, 0.0]}, "anchor_weight_schedule"),
    ({"rate_fd_mode": "forward"}, "rate_fd_mode"),
])
def test_removed_non_literal_flags_raise(inputs, over, name):
    with pytest.raises(ValueError, match=name):
        BM.SaeBlankfreeModelV1(**_kwargs(inputs, **over))


def test_class_trigram_history_raises(inputs):
    # the blank-free model forces the full trigram, so the base model is asked directly
    with pytest.raises(ValueError, match="class_trigram"):
        EM.SaeEmcModelV1(**_kwargs(inputs, prior_history="class_trigram", prior_order=3))


def test_zero_anchor_schedule_is_accepted(inputs):
    model = BM.SaeBlankfreeModelV1(**_kwargs(inputs, anchor_weight_schedule=[0.0, 0.0, 0.0]))
    assert model.anchor_weight(2) == 0.0


def test_removed_flags_at_their_defaults_are_accepted(inputs):
    stated = dict(BM.REMOVED_BLANKFREE_FLAGS)
    stated.update({k: v for k, v in EM.REMOVED_EMC_FLAGS.items() if k not in stated})
    BM.SaeBlankfreeModelV1(**_kwargs(inputs, **stated))
