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


# ===================================================================================================
# Priority-2 tests (test plan 2026-09-24, section 4): T2.1 train-step assembly and weights, T2.2 the
# configured constants of ctrl_20, T2.3 the model-side schedule reading, T2.4 the null recognizer.
#
# The oracle is tests/lattice_oracle.py.  It enumerates the train step's ACTUAL latent set, which is
# the SIL-split set (``sil_split=True``): the model's ``prior_history`` is built while the lattice
# topology is still "ctc" (model/emc_model.py:367), so a SIL run may split into several SIL tokens
# (the defect pinned by test_model_lattice.py T1.6).  With the model option ``sil_run_collapse=True``
# (arm ctrl_20_rc) the history is rebuilt from the blank-free cfg and the oracle enumerates the
# run-collapse set (``sil_split=False``) instead; :func:`_enum` reads the option off the model.
# The prior of every latent is the fitted
# ``log_tri`` rounded to float32 (the model's buffer dtype) and walked by the oracle's own history
# rows (BOS-padded, over tokens); ``PhoneNgramPrior.per_token_log_probs`` is never used
# (model/prior.py:246 double-counts one-token strings).  The only Z = 0 row of these batches is
# S = 1, which the matmul path detects (test_model_lattice.py::test_t1_5_production_z_zero_s1), so
# nothing here relies on ``_logmm``'s floor (model/lattice.py:673-674).
# ===================================================================================================

import math  # noqa: E402

import lattice_oracle as O  # noqa: E402

from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as PP  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as LAT  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model import rate_term as RT  # noqa: E402
from i6_experiments.users.wu.experiments.unsupervised_asr.model.lexlat_train import lexlat_lambda  # noqa: E402

#: test_lm_phone_prior.CORPUS (the T1.6 fixture text), copied
P2_CORPUS = ["<SIL> AA B <SIL>", "<SIL> AA B AA <SIL>", "B IY", "<SIL> IY IY B <SIL>"] * 30
P2_TAGS = ["u0", "u1", "u2"]
P2_ETA_DIM, P2_UNITS = 4, 20
P2_REVERSE = {"n_units": P2_UNITS, "eta_dim": P2_ETA_DIM, "d_model": 16, "d_ff": 16}
RHO_HZ = 9.6619373279
TAU20 = [8.0, 5.039684199579493, 3.174802103936399] + [2.0] * 17  # schedules.phase_schedules(20)[0]


@pytest.fixture(scope="module")
def p2_inputs(tmp_path_factory):
    """``(prior object, prior.npz, eta.npz)``: the T1.6 corpus prior (40 symbols, |h| 1681)."""
    tmp = tmp_path_factory.mktemp("p2")
    counts, _ = PP.count_ngrams([line.split() for line in P2_CORPUS])
    prior = PP.PhoneNgramPrior.from_counts(counts)
    prior_path, eta_path = str(tmp / "prior.npz"), str(tmp / "eta.npz")
    prior.save(prior_path)
    rng = np.random.RandomState(0)
    np.savez_compressed(eta_path, tags=np.array(P2_TAGS), eta=rng.randn(3, P2_ETA_DIM).astype(np.float32))
    return prior, prior_path, eta_path


def _p2_model(p2_inputs, seed=0, **over):
    _, prior_path, eta_path = p2_inputs
    kw = dict(
        temperature_schedule=[8.0, 2.0, 2.0], anchor_weight_schedule=0.0, lam_agg=0.1,
        count_ema_decay=0.99, band=25, prior_weight=1.0, prior_npz_path=prior_path,
        eta_table_path=eta_path, reverse_kwargs=dict(P2_REVERSE),
        lam_rate=3.0, rate_rho_hz=RHO_HZ, rate_fd_eps=0.25, rate_fd_mode="central",
        lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True,
    )
    kw.update(over)
    torch.manual_seed(seed)  # theta / phi init
    return BM.SaeBlankfreeModelV1(**kw)


def _p2_batch(s_lens, orig, tags, seed=7):
    """extern_data of a batch whose retained lengths are ``s_lens`` (features and units alike)."""
    b = len(s_lens)
    g = torch.Generator().manual_seed(seed)
    batch = Dim(b, name="batch")
    tdim = Dim(name="time", dimension=None, dyn_size_ext=Tensor(
        "t", raw_tensor=torch.tensor(s_lens, dtype=torch.int32), dims=[batch], dtype="int32"))
    sdim = Dim(name="units_time", dimension=None, dyn_size_ext=Tensor(
        "s", raw_tensor=torch.tensor(s_lens, dtype=torch.int32), dims=[batch], dtype="int32"))
    feat = Dim(1024, name="feat")
    data = TensorDict()
    data.data["features"] = Tensor("features", dims=[batch, tdim, feat], dtype="float16", feature_dim=feat,
                                   raw_tensor=torch.randn(b, max(s_lens), 1024, generator=g).half())
    units = torch.randint(0, P2_UNITS, (b, max(s_lens)), generator=g, dtype=torch.int32)
    data.data["units"] = Tensor("units", dims=[batch, sdim], dtype="int32",
                                sparse_dim=Dim(P2_UNITS, name="u"), raw_tensor=units)
    odim = Dim(1, name="o")
    data.data["original_length"] = Tensor(
        "original_length", dims=[batch, odim], dtype="int32",
        raw_tensor=torch.tensor([[o] for o in orig], dtype=torch.int32))
    tag = Tensor("seq_tag", dims=[batch], dtype="string")
    tag.raw_tensor = np.array(tags)
    data.data["seq_tag"] = tag
    return data, units.long()


def _p2_step(model, data, epoch, *, record_fd=False):
    """One train step with backward; records the recognizer output (grad retained), every
    ``lattice_loss`` call of the step and, optionally, every lattice call of the rate term."""
    rec = {"log_q": [], "l_tau": [], "fd": []}

    def hook(_module, _inputs, output):
        output.retain_grad()
        rec["log_q"].append(output)

    real_loss, real_fb = TS.lattice_loss, RT.lattice_forward_backward

    def loss_wrapper(log_q, seg_table, **kw):
        loss, out = real_loss(log_q, seg_table, **kw)
        rec["l_tau"].append(dict(log_q=log_q.detach().clone(), seg=seg_table.detach().clone(), kw=kw,
                                 out=out, loss=loss.detach().clone()))
        return loss, out

    def fb_wrapper(*args, **kw):
        rec["fd"].append(dict(kw))
        return real_fb(*args, **kw)

    handle = model.recognizer.register_forward_hook(hook)
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(TS, "lattice_loss", loss_wrapper)
            if record_fd:
                mp.setattr(RT, "lattice_forward_backward", fb_wrapper)
            model.train()
            torch.manual_seed(0)  # dropout masks
            rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=epoch)
            TS.train_step(model=model, extern_data=data)
            ctx = rf.get_run_ctx()
            total = ctx.total_loss()
            total = total.raw_tensor if hasattr(total, "raw_tensor") else total
            total.backward(retain_graph=True)  # T2.1 pulls theta's oracle gradient through the graph
    finally:
        handle.remove()
    rec["losses"] = ctx.losses
    rec["total"] = total.detach()
    return rec


def _loss_value(losses, name):
    return float(losses[name].loss.raw_tensor.double().sum())


def _tri32(prior) -> torch.Tensor:
    """The fitted log_tri rounded to float32 (the model buffer's dtype), as float64."""
    return torch.as_tensor(np.asarray(prior.log_tri, dtype=np.float32).astype(np.float64))


def _topo(model) -> O.Topology:
    c = model.lattice_cfg
    return O.Topology(c.n_phones, c.sil_id, c.band, c.d_min, c.d_max, c.d_max_sil, c.recognizer_stride)


_ENUMS = {}


def _enum(model, T, S, s_cols):
    """The train step's latent set: SIL-split readings, or run-collapse ones under the model option
    ``sil_run_collapse`` (see the section comment)."""
    sil_split = not model.sil_run_collapse
    key = (T, S, s_cols, sil_split)
    if key not in _ENUMS:
        _ENUMS[key] = O.enumerate_utterance(T, S, _topo(model), "trigram", s_cols=s_cols, sil_split=sil_split)
    return _ENUMS[key]


def _utt_terms(enum, log_q_b, seg_b, tri, tau, beta, tilt=0.0):
    ac, pr, rv = O.pieces(enum, log_q_b, seg_b, tri)
    return (ac + beta * pr + rv + tilt * enum.n_nonsil) / tau


def _run_counts(log_q_b: torch.Tensor):
    """``(uni [K], bi [K, K])`` expected run counts of one utterance by enumerating every frame path
    (plan T1.16), float64 and differentiable in ``log_q_b`` ([T, K])."""
    t_n, k_n = log_q_b.shape
    paths = torch.cartesian_prod(*[torch.arange(k_n)] * t_n).view(-1, t_n)
    c_uni = torch.zeros(len(paths), k_n, dtype=torch.float64)
    c_bi = torch.zeros(len(paths), k_n, k_n, dtype=torch.float64)
    for p, path in enumerate(paths.tolist()):
        runs, _ = O.run_collapse(path)
        for r in runs:
            c_uni[p, r] += 1
        for j, k in zip(runs[:-1], runs[1:]):
            c_bi[p, j, k] += 1
    prob = log_q_b.gather(1, paths.t()).sum(0).exp() if t_n else None
    return prob @ c_uni, torch.einsum("n,njk->jk", prob, c_bi)


def _agg_oracle(model, log_q64, t_lens):
    """BlankfreeAggLoss at its first step (hat = the normalised batch counts), plan T1.17(a)."""
    counts = [_run_counts(log_q64[b, :t]) for b, t in enumerate(t_lens)]
    uni = sum(c[0] for c in counts)
    bi = sum(c[1] for c in counts)
    floor = model.agg.cfg.floor
    text_uni, text_bi = model.agg.text_uni.double(), model.agg.text_bi.double()

    def kl(text, hat):
        return (text * (text.clamp(min=floor).log() - hat.clamp(min=floor).log())).sum()

    w = model.agg.cfg
    return w.unigram_weight * kl(text_uni, uni / uni.sum()) + w.bigram_weight * kl(text_bi, bi / bi.sum())


T21_S, T21_ORIG = [6, 5, 1], [11, 5, 4]
T21_EPOCH, T21_TAU = 2, 2.0


def _t2_1(p2_inputs, eps, **over):
    prior = p2_inputs[0]
    model = _p2_model(p2_inputs, rate_fd_eps=eps, **over)
    data, units = _p2_batch(T21_S, T21_ORIG, P2_TAGS)
    rec = _p2_step(model, data, T21_EPOCH)
    return model, prior, units, rec


def _t2_1_oracle(model, prior, units, rec, eps):
    """Per-utterance oracle values and the oracle gradients of the assembled total on log_q."""
    (call,) = rec["l_tau"]
    (log_q32,) = rec["log_q"]
    cfg, tau, beta = model.lattice_cfg, call["kw"]["temperature"], call["kw"]["prior_weight"]
    assert (tau, beta) == (T21_TAU, 1.0)
    tri = _tri32(prior)
    assert torch.equal(call["kw"]["prior_log_bi"], tri)  # the buffer is the float32-rounded log_tri
    seg = call["seg"]
    s_cols = seg.shape[-1]
    t_lens = [math.ceil(s / 3) for s in T21_S]
    assert call["kw"]["feat_lens"].tolist() == t_lens == [2, 2, 1]
    lq = log_q32.detach().double().clone().requires_grad_(True)
    rho = RHO_HZ / cfg.frame_rate_hz
    kept, log_z, e_n, rate_v, l_tau_terms, rate_terms = [], [], [], [], [], []
    for b, (T, S) in enumerate(zip(t_lens, T21_S)):
        enum = _enum(model, T, S, s_cols)
        if enum.n == 0:
            kept.append(False)
            log_z.append(None), e_n.append(None), rate_v.append(None)
            continue
        kept.append(True)
        terms = _utt_terms(enum, lq[b, :T], seg[b], tri, tau, beta)
        lz = torch.logsumexp(terms, 0)
        en = (torch.softmax(terms, 0) * enum.n_nonsil).sum()
        r = ((en / T21_ORIG[b] - rho) / rho) ** 2
        log_z.append(float(lz)), e_n.append(float(en)), rate_v.append(float(r))
        l_tau_terms.append(-lz / S)
        rate_terms.append(r)
    n_keep = sum(kept)
    l_tau = torch.stack(l_tau_terms).sum() / n_keep
    rate = torch.stack(rate_terms).sum() / n_keep
    agg = _agg_oracle(model, lq, t_lens)
    total = model.lam_tau * l_tau + model.lam_rate * rate + model.lam_agg * agg
    (g_exact,) = torch.autograd.grad(total, lq, retain_graph=True)
    # the FD surrogate the step differentiates, computed on the enumeration at the SAME eps
    (g_rest,) = torch.autograd.grad(model.lam_tau * l_tau + model.lam_agg * agg, lq, retain_graph=True)
    g_fd = g_rest.clone()
    with torch.no_grad():
        for b, (T, S) in enumerate(zip(t_lens, T21_S)):
            if not kept[b]:
                continue
            enum = _enum(model, T, S, s_cols)
            posts = []
            for tilt in (+eps, -eps):
                pi = torch.softmax(_utt_terms(enum, lq[b, :T], seg[b], tri, tau, beta, tilt), 0)
                post = torch.zeros(T * cfg.n_phones, dtype=torch.float64)
                post.index_add_(0, enum.q_idx.reshape(-1), pi.repeat_interleave(T))
                posts.append(post.view(T, -1))
            g = (posts[0] - posts[1]) / (2 * eps)
            r = e_n[b] / T21_ORIG[b]
            coef = 2 * (r - rho) / (rho * rho * T21_ORIG[b])
            g_fd[b, :T] += model.lam_rate * coef * g / n_keep
    return dict(kept=kept, n_keep=n_keep, log_z=log_z, e_n=e_n, rate_v=rate_v, l_tau=float(l_tau),
                rate=float(rate), agg=float(agg), total=float(total), g_exact=g_exact, g_fd=g_fd,
                g_rest=g_rest,
                rho=rho, lq=lq.detach(), seg=seg, tri=tri, tau=tau, beta=beta, t_lens=t_lens)


def _rel(a, b):
    return abs(a - b) / max(1.0, abs(b))


@pytest.mark.parametrize("eps", [1e-4, 0.25])
def test_t2_1_train_step_assembly(p2_inputs, eps):
    """T2.1: l_tau and rate are means over the kept rows, agg is the T1.16/T1.17 oracle, the total is
    1 l_tau + 3 rate + 0.1 agg, and theta's gradient is the oracle gradient of that sum."""
    _check_t2_1_assembly(p2_inputs, eps)


@pytest.mark.parametrize("eps", [1e-4, 0.25])
def test_t2_1_rc_train_step_assembly(p2_inputs, eps):
    """T2.1 with ``sil_run_collapse=True`` (ctrl_20_rc): the same assembly against the run-collapse
    oracle, with the step's history the model's rebuilt blank-free one."""
    model, rec = _check_t2_1_assembly(p2_inputs, eps, sil_run_collapse=True)
    (call,) = rec["l_tau"]
    assert model.sil_run_collapse is True and call["kw"]["history"] is model.prior_history
    rebuilt = LAT.build_prior_history(model.lattice_cfg, "trigram")
    assert torch.equal(model.prior_history.same_nonsil, rebuilt.same_nonsil)
    # the oracle's latent set is not the SIL-split one on either kept row
    topo, s_cols = _topo(model), call["seg"].shape[-1]
    for b in (0, 1):
        T = math.ceil(T21_S[b] / 3)
        split = O.enumerate_utterance(T, T21_S[b], topo, "trigram", s_cols=s_cols, sil_split=True)
        assert _enum(model, T, T21_S[b], s_cols).n < split.n, b


def _check_t2_1_assembly(p2_inputs, eps, **over):
    """The body of :func:`test_t2_1_train_step_assembly`; ``over`` are extra model keywords."""
    model, prior, units, rec = _t2_1(p2_inputs, eps, **over)
    o = _t2_1_oracle(model, prior, units, rec, eps)
    losses, (call,) = rec["losses"], rec["l_tau"]
    out = call["out"]
    assert (model.lam_tau, model.lam_rate, model.lam_agg) == (1.0, 3.0, 0.1)
    # the kept rows: S = 1 is infeasible and detected; n_keep = 2
    assert o["kept"] == [True, True, False] and o["n_keep"] == 2
    assert out.z_zero.tolist() == [False, False, True]
    for b in (0, 1):
        assert _rel(float(out.log_z[b]), o["log_z"][b]) <= 1e-10, (b, float(out.log_z[b]), o["log_z"][b])
        assert abs(float(RT.expected_nonsil_tokens(out, model.lattice_cfg)[b]) - o["e_n"][b]) <= 1e-10
    # the three trained terms and their scales
    assert (losses["l_tau"].scale, losses["rate"].scale, losses["agg"].scale) == (1.0, 3.0, 0.1)
    assert not any(losses[k].as_error for k in ("l_tau", "rate", "agg"))
    l_tau, rate, agg = (_loss_value(losses, k) for k in ("l_tau", "rate", "agg"))
    print(f"T2.1 eps={eps}: l_tau {l_tau!r} oracle {o['l_tau']!r}; rate {rate!r} oracle {o['rate']!r}; "
          f"agg {agg!r} oracle {o['agg']!r}; total {float(rec['total'])!r} oracle {o['total']!r}")
    assert _rel(l_tau, o["l_tau"]) <= 1e-10
    assert _rel(rate, o["rate"]) <= 1e-10
    assert abs(agg - o["agg"]) <= 1e-5 * max(1.0, abs(o["agg"]))  # agg runs on the float32 log_q
    # RETURNN's total = sum of scale x value over the trained losses; agg is a float32 tensor, so its
    # 0.1 x agg is rounded in float32 before the float64 sum (hence 1e-7, not 1e-12)
    assert abs(float(rec["total"]) - (1.0 * l_tau + 3.0 * rate + 0.1 * agg)) <= 1e-7 * max(1.0, abs(o["total"]))
    assert abs(float(rec["total"]) - o["total"]) <= 1e-6 * max(1.0, abs(o["total"]))
    # the monitor of the rate: 50 * mean_kept(E[N] / original)
    hz = 50.0 * np.mean([o["e_n"][b] / T21_ORIG[b] for b in (0, 1)])
    assert _rel(_loss_value(losses, "blankfree_expected_phone_rate_hz"), hz) <= 1e-6
    assert _rel(_loss_value(losses, "blankfree_z_zero_frac"), 1.0 / 3.0) <= 1e-7
    # theta.  log_q = log_softmax(logits), so what reaches theta is the gradient on the LOGITS,
    # g - q sum_c g (the simplex projection).  The raw log_q gradients of two functions that agree
    # only on the simplex differ off it: agg's closed-form run counts q_t(k)(1 - q_{t-1}(k)) assume
    # sum_c q = 1 while the enumeration does not, and they differ by lambda_t q_t, which the
    # projection removes.  So the logit gradients are compared, and theta's parameter gradients are
    # the oracle output gradient pulled back through the recognizer's own graph.
    (log_q,) = rec["log_q"]
    q = log_q.detach().double().exp()

    def proj(g):
        return g - q * g.sum(-1, keepdim=True)

    grad = proj(log_q.grad.double())
    g_fd, g_exact = proj(o["g_fd"]), proj(o["g_exact"])
    scale = float(g_exact.abs().max())
    d_fd = float((grad - g_fd).abs().max())
    d_exact = float((grad - g_exact).abs().max())
    gap = float((g_fd - g_exact).abs().max())
    rate_scale = float(proj(o["g_exact"] - o["g_rest"]).abs().max())
    print(f"T2.1 eps={eps}: logit grads: max|step - oracle FD surrogate| = {d_fd:.3e}; "
          f"max|step - exact oracle| = {d_exact:.3e} (max|exact| {scale:.3e}, relative {d_exact / scale:.3e}; "
          f"relative to the rate part's max {rate_scale:.3e}: {d_exact / rate_scale:.3e}); "
          f"oracle FD-vs-exact gap {gap:.3e}")
    assert d_fd <= 1e-6 * max(1.0, scale)
    if eps == 1e-4:
        assert d_exact <= 1e-6 * max(1.0, scale)
    params = [p for p in model.recognizer.parameters() if p.requires_grad]
    assert all(p.grad is not None for p in params)
    pulled = torch.autograd.grad(log_q, params, grad_outputs=o["g_fd"].to(log_q.dtype))
    worst = 0.0
    for p, g in zip(params, pulled):
        s = max(1.0, float(g.abs().max()))
        worst = max(worst, float((p.grad - g).abs().max()) / s)
    print(f"T2.1 eps={eps}: theta parameter grads vs the oracle FD-surrogate gradient pulled back: "
          f"max scaled deviation {worst:.3e}")
    assert worst <= 1e-6
    return model, rec


def test_t2_1_reverse_gradient_is_l_tau_only(p2_inputs):
    """T2.1: phi's gradient equals a run with lam_rate = lam_agg = 0 (bitwise) and the oracle
    gradient of the l_tau mean alone (1e-6)."""
    _check_t2_1_reverse_gradient(p2_inputs)


def test_t2_1_rc_reverse_gradient_is_l_tau_only(p2_inputs):
    """:func:`test_t2_1_reverse_gradient_is_l_tau_only` with ``sil_run_collapse=True`` (both runs)."""
    _check_t2_1_reverse_gradient(p2_inputs, sil_run_collapse=True)


def _check_t2_1_reverse_gradient(p2_inputs, **over):
    model, prior, units, rec = _t2_1(p2_inputs, 0.25, **over)
    model0, _, _, rec0 = _t2_1(p2_inputs, 0.25, lam_rate=0.0, lam_agg=0.0, **over)
    assert "rate" not in [k for k, v in rec0["losses"].items() if v.scale != 0.0 and not v.as_error]
    names = [n for n, p in model.reverse.named_parameters() if p.requires_grad]
    g_full = dict((n, p.grad) for n, p in model.reverse.named_parameters() if p.requires_grad)
    g_zero = dict((n, p.grad) for n, p in model0.reverse.named_parameters() if p.requires_grad)
    for n in names:
        assert g_full[n] is not None and g_zero[n] is not None, n
        assert torch.equal(g_full[n], g_zero[n]), (n, float((g_full[n] - g_zero[n]).abs().max()))
    # oracle: autograd of mean_kept(-log Z / S) through build_segment_table
    (call,) = rec["l_tau"]
    (log_q32,) = rec["log_q"]
    tri, tau = _tri32(prior), call["kw"]["temperature"]
    eta = model.lookup_eta(P2_TAGS, "cpu")
    seg = LAT.build_segment_table(model.reverse, units, eta).double()
    assert torch.equal(seg.detach(), call["seg"])
    lq = log_q32.detach().double()
    terms = []
    for b, S in enumerate(T21_S):
        T = math.ceil(S / 3)
        enum = _enum(model, T, S, seg.shape[-1])
        if enum.n == 0:
            continue
        terms.append(-torch.logsumexp(_utt_terms(enum, lq[b, :T], seg[b], tri, tau, 1.0), 0) / S)
    l_tau = torch.stack(terms).mean()
    params = [p for p in model.reverse.parameters() if p.requires_grad]
    oracle = torch.autograd.grad(l_tau, params, allow_unused=True)
    worst = 0.0
    for n, p, g in zip(names, params, oracle):
        g = torch.zeros_like(p) if g is None else g
        scale = max(1.0, float(g.abs().max()))
        d = float((p.grad.double() - g.double()).abs().max())
        worst = max(worst, d / scale)
        assert d <= 1e-6 * scale, (n, d)
    print(f"T2.1 reverse grads vs l_tau-only oracle{' (rc)' if over else ''}: max scaled deviation {worst:.3e}")


# --- sil_run_collapse (arm ctrl_20_rc): the history is the only thing the option moves -----------


def test_sil_run_collapse_moves_only_the_history(p2_inputs):
    """Default: the history the base class built under the "ctc" topology (the banked behaviour, table
    for table).  Option on: the one built from the blank-free cfg.  Same seed, same parameters,
    buffers and lattice cfg, so a checkpoint loads into either."""
    from dataclasses import replace

    default = _p2_model(p2_inputs)
    rc = _p2_model(p2_inputs, sil_run_collapse=True)
    assert default.sil_run_collapse is False and rc.sil_run_collapse is True
    assert default.lattice_cfg == rc.lattice_cfg and rc.lattice_cfg.topology == "blankfree"
    fields = ("succ", "last", "next", "group", "members", "dst", "same_nonsil")
    for model, cfg in ((default, replace(default.lattice_cfg, topology="ctc")), (rc, rc.lattice_cfg)):
        ref = LAT.build_prior_history(cfg, "trigram")
        assert (model.prior_history.name, model.prior_history.n_outer, model.prior_history.start) == (
            ref.name, ref.n_outer, ref.start)
        for f in fields:
            assert torch.equal(getattr(model.prior_history, f), getattr(ref, f)), f
    sd_d, sd_r = default.state_dict(), rc.state_dict()
    assert list(sd_d) == list(sd_r)
    assert all(torch.equal(sd_d[k], sd_r[k]) for k in sd_d)


# --- T2.2 the configured constants of ctrl_20 ----------------------------------------------------


def _graph_data():
    from sisyphus import tk

    p = lambda name: tk.Path(f"/nonexistent/{name}")  # noqa: E731
    return dict(
        train_feature_hdfs=[p("feats.0.hdf")], train_units_hdfs=[p("units.0.hdf")],
        train_original_hdfs=[p("orig.0.hdf")], dev_feature_hdfs=[p("feats.0.hdf")],
        dev_units_hdfs=[p("units.0.hdf")], dev_original_hdfs=[p("orig.0.hdf")],
        train_segments=p("train.segments"), dev_segments=p("cv.segments"), prior_npz=p("prior.npz"),
        eta_npz=p("eta.npz"), flat_checkpoint=p("flat_init.pt"),
    )


def test_t2_2_ctrl_20_constants(tmp_path, p2_inputs):
    from i6_experiments.common.setups.returnn_pytorch.serialization import Collection

    from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms, jobs

    arm = arms.ctrl_20(data=_graph_data())
    args = dict(jobs.get_model_args(arm.returnn_config))
    expected = dict(lam_agg=0.1, lam_rate=3.0, count_ema_decay=0.99, band=25, prior_weight=1.0,
                    rate_fd_eps=0.25, rate_fd_mode="central", rate_rho_hz=9.6619373279,
                    lattice_reduction="matmul", lattice_checkpoint=32, lattice_float64=True,
                    anchor_weight_schedule=0.0)
    for k, v in expected.items():
        assert args[k] == v and type(args[k]) is type(v), (k, args[k], v)
    assert args["temperature_schedule"] == TAU20
    assert "lam_tau" not in args and "prior_history" not in args  # the model's defaults / forcing
    (coll,) = [o for o in arm.returnn_config.python_prolog if isinstance(o, Collection)]
    (groups,) = [o for o in coll.serializer_objects if getattr(o, "import_as", None) == "emc_param_groups"]
    assert groups.hashed_arguments == {"recognizer_lr_multiplier": 1.0, "reverse_lr_multiplier": 30.0}
    # the model these arguments build (the two init/input paths replaced by test fixtures; the
    # recognizer init checkpoint is T2.12's)
    _, prior_path, _ = p2_inputs
    eta_path = str(tmp_path / "eta16.npz")
    np.savez_compressed(eta_path, tags=np.array(P2_TAGS), eta=np.zeros((3, 16), np.float32))
    args.update(prior_npz_path=prior_path, eta_table_path=eta_path)
    args.pop("recognizer_checkpoint_path")
    model = BM.SaeBlankfreeModelV1(**args)
    assert model.lam_tau == 1.0 and model.lam_agg == 0.1 and model.lam_rate == 3.0
    assert model.agg_cfg.count_ema_decay == 0.99 and model.agg.cfg.count_ema_decay == 0.99
    assert (model.rate_fd_eps, model.rate_fd_mode, model.rate_rho_hz) == (0.25, "central", 9.6619373279)
    assert (model.lattice_reduction, model.lattice_checkpoint, model.lattice_float64) == ("matmul", 32, True)
    assert model.prior_weight_at(1) == 1.0 and model.prior_weight == 1.0
    assert all(model.anchor_weight(e) == 0.0 for e in range(1, 21))
    assert model.prior_history_name == "trigram" and model.prior_history.name == "trigram"
    assert model.prior_history.n_hist == 41 * 41 and tuple(model.prior_log_bi.shape) == (1681, 40)
    c = model.lattice_cfg
    assert (c.topology, c.recognizer_stride, c.band, c.d_min, c.d_max, c.d_max_sil, c.n_phones, c.sil_id) == (
        "blankfree", 3, 25, 2, 25, 50, 40, 39)
    assert (model.recognizer.n_out, model.recognizer.stride, model.recognizer.in_dim) == (40, 3, 1024)
    pg = PG.emc_param_groups(model=model, **groups.hashed_arguments)
    assert [g["learning_rate_multiplier"] for g in pg] == [1.0, 30.0]
    assert arm.returnn_config.config["learning_rate"] * 30.0 == 3.0e-3


# --- T2.3 the model-side schedule reading ----------------------------------------------------------


def _rampout_k2_model(p2_inputs):
    from i6_experiments.users.wu.experiments.unsupervised_asr.training import schedules as SCH

    tau, _ = SCH.phase_schedules(20)
    beta = SCH.phone_trigram_weight_schedule("rampout", 20, onset=8, ramp=3, full_lam=1.0)
    # the k2 runtime is built from its arguments alone (no file is read before its first step, which
    # this test replaces by a recorder); onset / ramp / full_lam are the arms' K2_ONSET / 3 / 1.0
    return _p2_model(p2_inputs, temperature_schedule=tau, prior_weight_schedule=beta,
                     lexlat_k2_hlg="/nonexistent/HLG.pt", lexlat_k2_stats="/nonexistent/build.json",
                     lexlat_k2_max_active=1000, lexlat_k2_onset=8, lexlat_k2_ramp=3,
                     lexlat_k2_full_lam=1.0), tau, beta


def test_t2_3_temperature_and_beta_tables(p2_inputs):
    model, tau, beta = _rampout_k2_model(p2_inputs)
    assert tau == TAU20
    for e in range(1, 26):
        assert model.temperature(e) == TAU20[min(e, 20) - 1], e
        assert model.prior_weight_at(e) + lexlat_lambda(e, onset=8, ramp=3, full=1.0) == 1.0, e
        assert model.lexlat_k2.lam(e) == lexlat_lambda(e, onset=8, ramp=3, full=1.0)
        assert model.lexlat_k2.active(e) == (e >= 8)


def test_t2_3_e60_learning_rates():
    from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms, jobs

    arm = arms.ctrl_20(data=_graph_data(), num_sub_epochs=60)
    lr = arm.returnn_config.config["learning_rates"]
    assert len(lr) == 60 and lr[19] == 1e-5 and lr[20] == 1e-4 and lr[59] == 1e-4
    tau = jobs.get_model_args(arm.returnn_config)["temperature_schedule"]
    assert tau[:20] == TAU20 and tau[20:] == [2.0] * 40


@pytest.mark.parametrize("epoch", [1, 4, 8, 9, 10, 20])
def test_t2_3_step_reads_the_schedules(p2_inputs, epoch):
    """The l_tau pass and every rate-term pass run at tau(e) and beta(e); the k2 step is called only
    from its on-set, at tau(e), and its loss is marked at lam_lex(e)."""
    model, tau, beta = _rampout_k2_model(p2_inputs)
    calls = []

    def fake_k2_step(log_q, *, feat_lens, retained, keep, epoch, temperature, cfg, global_step):
        calls.append(dict(epoch=epoch, temperature=temperature, dtype=log_q.dtype))
        return log_q.sum() * 0.0, {}

    model.lexlat_k2.step = fake_k2_step
    data, _ = _p2_batch([6, 5], [11, 5], P2_TAGS[:2])
    rec = _p2_step(model, data, epoch, record_fd=True)
    (call,) = rec["l_tau"]
    t_e, b_e = TAU20[epoch - 1], beta[epoch - 1]
    assert call["kw"]["temperature"] == t_e and call["kw"]["prior_weight"] == b_e
    assert rec["fd"], "the rate term ran no lattice pass"
    for kw in rec["fd"]:
        assert kw["temperature"] == t_e and kw["prior_weight"] == b_e
        assert kw["history"] is call["kw"]["history"] and kw["cfg"] is call["kw"]["cfg"]
    assert _loss_value(rec["losses"], "blankfree_temperature") == pytest.approx(t_e, rel=1e-7)
    assert _loss_value(rec["losses"], "prior_weight_eff") == b_e
    if epoch < 8:
        assert calls == [] and "lexlat_k2" not in rec["losses"]
    else:
        assert calls == [dict(epoch=epoch, temperature=t_e, dtype=torch.float64)]
        assert rec["losses"]["lexlat_k2"].scale == lexlat_lambda(epoch, onset=8, ramp=3, full=1.0)


# --- T2.4 the null recognizer ---------------------------------------------------------------------


_NULL_F32 = (
    "FINDING model/blankfree_model.py:617 (null_log_q): the null recognizer is float32 -log 40 "
    "(-3.68887949 vs -3.6888794541), so the null l_tau_b S_b equals (T_rec / tau) x 3.68887949 - "
    "log Z(log_q = 0), not the plan's / genmarg's (T_rec / tau) log 40 (reverse_model/genmarg_steps.py:287 "
    "uniform_q_offset in float64): the two differ by T_rec x 3.6e-8 / tau nats per utterance"
)


def _t2_4(p2_inputs):
    from i6_experiments.users.wu.experiments.unsupervised_asr.reverse_model import genmarg_steps as GS

    prior = p2_inputs[0]
    model = _p2_model(p2_inputs, null_recognizer=True, freeze_recognizer=True)
    data, units = _p2_batch(T21_S, T21_ORIG, P2_TAGS)
    rec = _p2_step(model, data, T21_EPOCH)
    (call,) = rec["l_tau"]
    tau, tri = call["kw"]["temperature"], _tri32(prior)
    zero = torch.zeros(3, 2, 40, dtype=torch.float64)
    log_z0 = []
    for b, S in enumerate(T21_S):
        T = math.ceil(S / 3)
        enum = _enum(model, T, S, call["seg"].shape[-1])
        log_z0.append(None if enum.n == 0 else float(
            torch.logsumexp(_utt_terms(enum, zero[b, :T], call["seg"][b], tri, tau, 1.0), 0)))
    res = GS.generative_marginals(model, units, torch.tensor(T21_S), P2_TAGS, rate=False)
    rows = GS.marginal_rows(res, torch.tensor(T21_S), P2_TAGS)
    return model, rec, call, tau, log_z0, rows


def test_t2_4_null_recognizer(p2_inputs):
    """The null step: agg / rate are errors, theta has no gradient, and l_tau_b S_b = (T_rec / tau) c
    - log Z(log_q = 0) (1e-10) with c = -null_log_q, the float32 rounding of log 40 (the exact log 40
    of the plan is the strict xfail below)."""
    model, rec, call, tau, log_z0, rows = _t2_4(p2_inputs)
    assert rec["log_q"] == []  # the recognizer never ran
    c32 = -float(torch.tensor(-math.log(40), dtype=torch.float32))
    # null_log_q is float32 -log 40, up-cast by lattice_float64
    assert torch.equal(call["log_q"], torch.full(call["log_q"].shape, -c32, dtype=torch.float64))
    losses = rec["losses"]
    assert losses["agg"].as_error and losses["rate"].as_error and not losses["l_tau"].as_error
    assert "blankfree_rate_fd_check" not in losses
    assert all(p.grad is None for p in model.recognizer.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in model.reverse.parameters())
    per_utt = []
    for b, S in enumerate(T21_S):
        T = math.ceil(S / 3)
        if log_z0[b] is None:
            assert bool(call["out"].z_zero[b]) and float(call["loss"][b]) == 0.0 and rows[b]["impossible"]
            continue
        got = float(call["loss"][b])  # = l_tau_b * S_b
        expected = (T / tau) * c32 - log_z0[b]
        print(f"T2.4 utt {b}: S={S} T_rec={T} l_tau*S {got!r} oracle(float32 log 40) {expected!r} "
              f"oracle(exact log 40) {(T / tau) * math.log(40) - log_z0[b]!r}")
        assert _rel(got, expected) <= 1e-10
        per_utt.append(got / S)
        # genmarg's zero-log_q value is the same log Z(log_q = 0)
        assert _rel(rows[b]["l_tau2"], -log_z0[b]) <= 1e-10
        assert rows[b]["recognizer_frames"] == T
        assert rows[b]["uniform_q_offset_tau2"] == T * math.log(40) / 2.0
    assert _rel(_loss_value(losses, "l_tau"), sum(per_utt) / len(per_utt)) <= 1e-12


@pytest.mark.xfail(strict=True, reason=_NULL_F32)
def test_t2_4_null_offset_is_exact_log_40(p2_inputs):
    """The plan's identity with the exact constant: l_tau_b S_b = (T_rec / tau) log 40 - log Z(0)
    = genmarg's l_tau2 + uniform_q_offset_tau2 (1e-10)."""
    _, _, call, tau, log_z0, rows = _t2_4(p2_inputs)
    for b, S in enumerate(T21_S):
        if log_z0[b] is None:
            continue
        T = math.ceil(S / 3)
        got = float(call["loss"][b])
        assert _rel(got, (T / tau) * math.log(40) - log_z0[b]) <= 1e-10, b
        assert _rel(got, rows[b]["l_tau2"] + rows[b]["uniform_q_offset_tau2"]) <= 1e-10, b


# --- T2.5 the objective's constants pinned to the method, not to the code's config -----------------
#
# Review 2026-09-24 (reports/review_test_oracles_2026-09-24.md, findings 1-3): T2.1's oracle reads
# the rate clock (``cfg.frame_rate_hz``) and the agg weights (``model.agg.cfg``) from the code, and
# T2.3's fake k2 step records neither ``retained`` nor ``keep`` nor ``feat_lens``, so a wrong clock,
# a changed agg weight or ``retained = original`` passed the whole suite.  Every constant below is a
# LITERAL taken from the method, with its source:
#
# * 50 Hz: the features and the rate term's T_audio are on the 50 Hz clock of the ORIGINAL audio
#   (SAE_i6_ref_objective.md 4.2; SAE_i6_ref_blankfree.md section 1: "features ... 50 Hz", "rho =
#   9.6619373279 phones per original-audio second", "T_audio is the original duration").
# * agg weights 1 and 1: SAE_4A.md:126-127 (the JUPITER pre-registration of 2026-09-15) fixes
#   "lambda_agg = 0.1 on KL(c_text || c_hat) summed over unigram and bigram terms", i.e. an
#   unweighted sum; SAE_i6_ref_objective.md 4.3 and SAE_i6_ref_blankfree.md section 1 (L_agg row)
#   define the same unweighted KL.  No JUPITER config or log states a separate per-order weight
#   (SAE_4A_blankfree.md "Registered first model" and SAE_4A_attrib.md do not mention one).
# * the k2 term: SAE_i6_ref_lexicon.md B4, "lam_lex x (-L_lex) / n_unit_frames", n_unit_frames the
#   retained (VAD-kept) frame count S, the lattice term's own divisor; curriculum on-set 8, ramp 3,
#   full_lam 1, "lam = 1/3, 2/3 and 1 at sub-epochs 8, 9 and 10" (the D15 log, i.e. what ran).

FRAME_RATE_HZ_METHOD = 50.0
AGG_UNIGRAM_WEIGHT_METHOD, AGG_BIGRAM_WEIGHT_METHOD = 1.0, 1.0
LAM_LEX_METHOD = {7: 0.0, 8: 1.0 / 3.0, 9: 2.0 / 3.0, 10: 1.0, 20: 1.0}


def _bed_model(tmp_path, p2_inputs, arm, seed=0):
    """The model an arm's own ``get_model`` arguments build (T2.2's construction): the prior / eta
    paths replaced by the fixtures, the recognizer init checkpoint dropped."""
    from i6_experiments.users.wu.experiments.unsupervised_asr.training import jobs

    args = dict(jobs.get_model_args(arm.returnn_config))
    _, prior_path, _ = p2_inputs
    eta_path = str(tmp_path / "eta16.npz")
    np.savez_compressed(eta_path, tags=np.array(P2_TAGS), eta=np.zeros((3, 16), np.float32))
    args.update(prior_npz_path=prior_path, eta_table_path=eta_path)
    args.pop("recognizer_checkpoint_path")
    torch.manual_seed(seed)
    return BM.SaeBlankfreeModelV1(**args)


def _ctrl20_model(tmp_path, p2_inputs, seed=0):
    from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms

    return _bed_model(tmp_path, p2_inputs, arms.ctrl_20(data=_graph_data()), seed)


def _k2lat_model(tmp_path, p2_inputs, seed=0):
    from sisyphus import tk

    from i6_experiments.users.wu.experiments.unsupervised_asr.training import arms

    graph = dict(hlg=tk.Path("/nonexistent/HLG.pt"), stats=tk.Path("/nonexistent/build.json"),
                 resources=tk.Path("/nonexistent/trie.npz"),
                 expected_build=dict(backoff_loops="word_boundary", escape=True, sil_prob=0.5, theta=0.0))
    return _bed_model(tmp_path, p2_inputs, arms.k2lat_20_ma3000(data=_graph_data(), graph=graph), seed)


def _expected_nonsil_oracle(model, prior, rec, s_lens):
    """E[N] per utterance (None on an infeasible row) by enumeration of the step's latent set."""
    (call,) = rec["l_tau"]
    (log_q32,) = rec["log_q"]
    tau, beta = call["kw"]["temperature"], call["kw"]["prior_weight"]
    tri, seg = _tri32(prior), call["seg"]
    lq = log_q32.detach().double()
    e_n = []
    for b, S in enumerate(s_lens):
        T = math.ceil(S / 3)
        enum = _enum(model, T, S, seg.shape[-1])
        if enum.n == 0:
            e_n.append(None)
            continue
        terms = _utt_terms(enum, lq[b, :T], seg[b], tri, tau, beta)
        e_n.append(float((torch.softmax(terms, 0) * enum.n_nonsil).sum()))
    return e_n


def test_t2_5_ctrl_20_rate_clock_is_50_hz(tmp_path, p2_inputs):
    """Finding 1: the frame rate the rate term divides rho by is 50 (the method's clock), on the
    model ctrl_20's own arguments build."""
    model = _ctrl20_model(tmp_path, p2_inputs)
    # train_step.py: rho = model.rate_rho_hz / model.lattice_cfg.frame_rate_hz
    assert model.lattice_cfg.frame_rate_hz == FRAME_RATE_HZ_METHOD
    assert model.rate_rho_hz == RHO_HZ


def test_t2_5_rate_term_follows_original_length_at_50_hz(tmp_path, p2_inputs):
    """Finding 1, end to end: the same batch under two original lengths.  E[N] does not move, and
    the step's rate term is mean_kept(((E[N] / (orig / 50) - rho) / rho)^2) with 50 and rho the
    method's literals, in both runs and in their difference."""
    prior = p2_inputs[0]
    origs = ([11, 5, 4], [23, 9, 4])
    got, want, e_ns = [], [], []
    for orig in origs:
        model = _ctrl20_model(tmp_path, p2_inputs)
        data, _ = _p2_batch(T21_S, orig, P2_TAGS)
        rec = _p2_step(model, data, T21_EPOCH)
        assert rec["l_tau"][0]["kw"]["temperature"] == TAU20[T21_EPOCH - 1]
        e_n = _expected_nonsil_oracle(model, prior, rec, T21_S)
        assert e_n[2] is None and e_n[0] is not None and e_n[1] is not None  # S = 1 is infeasible
        oracle = np.mean([((e_n[b] / (orig[b] / 50.0) - RHO_HZ) / RHO_HZ) ** 2 for b in (0, 1)])
        got.append(_loss_value(rec["losses"], "rate"))
        want.append(float(oracle))
        e_ns.append(e_n)
        print(f"T2.5 rate: orig {orig}: E[N] {e_n[:2]} step {got[-1]!r} oracle(50 Hz) {oracle!r}")
    assert all(abs(a - b) <= 1e-12 for a, b in zip(e_ns[0][:2], e_ns[1][:2]))  # only orig changed
    for g, w in zip(got, want):
        assert _rel(g, w) <= 1e-10, (g, w)
    assert abs(want[0] - want[1]) > 1e-3  # the change is material on this instance
    assert abs((got[0] - got[1]) - (want[0] - want[1])) <= 1e-10 * max(1.0, abs(want[0] - want[1]))


def test_t2_5_agg_weights_are_the_bed_s(tmp_path, p2_inputs):
    """Finding 2: L_agg = 1 x KL(unigram) + 1 x KL(bigram) (SAE_4A.md:126-127), on the model ctrl_20
    builds; the weights are literals here, the targets the model's own (S9, T1.17)."""
    model = _ctrl20_model(tmp_path, p2_inputs)
    assert (model.agg.cfg.unigram_weight, model.agg.cfg.bigram_weight) == (
        AGG_UNIGRAM_WEIGHT_METHOD, AGG_BIGRAM_WEIGHT_METHOD)
    data, _ = _p2_batch(T21_S, T21_ORIG, P2_TAGS)
    rec = _p2_step(model, data, T21_EPOCH)
    (log_q32,) = rec["log_q"]
    t_lens = [math.ceil(s / 3) for s in T21_S]
    lq = log_q32.detach().double()
    counts = [_run_counts(lq[b, :t]) for b, t in enumerate(t_lens)]
    uni, bi = sum(c[0] for c in counts), sum(c[1] for c in counts)
    floor = model.agg.cfg.floor
    text_uni, text_bi = model.agg.text_uni.double(), model.agg.text_bi.double()

    def kl(text, hat):
        return float((text * (text.clamp(min=floor).log() - hat.clamp(min=floor).log())).sum())

    kl_uni, kl_bi = kl(text_uni, uni / uni.sum()), kl(text_bi, bi / bi.sum())
    oracle = AGG_UNIGRAM_WEIGHT_METHOD * kl_uni + AGG_BIGRAM_WEIGHT_METHOD * kl_bi
    agg = _loss_value(rec["losses"], "agg")
    print(f"T2.5 agg: step {agg!r} oracle {oracle!r} (KL uni {kl_uni!r}, KL bi {kl_bi!r})")
    assert kl_uni > 1e-2 and kl_bi > 1e-2  # a weight change moves agg far beyond the tolerance
    assert abs(agg - oracle) <= 1e-5 * max(1.0, abs(oracle))  # agg runs on the float32 log_q


#: a stand-in -L_lex >= 0 per utterance for the recording k2 step (distinct, so a mis-assigned row
#: or divisor shows)
_NEG_L_LEX = [0.7, 1.3, 2.9]


@pytest.mark.parametrize("epoch", sorted(LAM_LEX_METHOD))
def test_t2_5_train_step_k2_plumbing(tmp_path, p2_inputs, epoch):
    """Finding 3: on the k2lat_20_ma3000 model, the train step hands the k2 step retained = S (not
    original_length, not T), keep = the kept-row mask, feat_lens = ceil(S / 3), this sub-epoch and
    tau; the added loss is lam_lex(epoch) x term with B4's curriculum, and term is B4's
    mean_kept[(-L_lex) / S] of the rows the step handed over."""
    model = _k2lat_model(tmp_path, p2_inputs)
    calls = []

    def recording_k2_step(log_q, *, feat_lens, retained, keep, epoch, temperature, cfg, global_step):
        calls.append(dict(feat_lens=feat_lens.detach().clone(), retained=retained.detach().clone(),
                          keep=keep.detach().clone(), epoch=epoch, temperature=temperature,
                          shape=tuple(log_q.shape)))
        neg = torch.tensor(_NEG_L_LEX, dtype=log_q.dtype)
        k = keep.to(log_q.dtype)
        term = ((neg / retained.to(log_q.dtype)) * k).sum() / k.sum().clamp(min=1)
        return term + 0.0 * log_q.sum(), {}

    model.lexlat_k2.step = recording_k2_step
    data, _ = _p2_batch(T21_S, T21_ORIG, P2_TAGS)
    rec = _p2_step(model, data, epoch)
    losses = rec["losses"]
    lam = LAM_LEX_METHOD[epoch]
    if lam == 0.0:
        assert calls == [] and "lexlat_k2" not in losses
        return
    (c,) = calls
    s_lit = [float(s) for s in T21_S]  # the retained (VAD-kept) frame counts of the batch
    assert c["retained"].tolist() == s_lit, c["retained"]
    assert c["retained"].tolist() != [float(o) for o in T21_ORIG]  # not original_length
    assert c["feat_lens"].tolist() == [math.ceil(s / 3) for s in T21_S] == [2, 2, 1]
    assert c["keep"].tolist() == [1.0, 1.0, 0.0]  # S = 1 is infeasible (z_zero), rows 0, 1 kept
    assert c["keep"].tolist() == (~rec["l_tau"][0]["out"].z_zero).double().tolist()
    assert (c["epoch"], c["temperature"]) == (epoch, TAU20[epoch - 1])
    assert c["shape"][:2] == (3, 2)
    term_oracle = (_NEG_L_LEX[0] / T21_S[0] + _NEG_L_LEX[1] / T21_S[1]) / 2.0
    assert losses["lexlat_k2"].scale == pytest.approx(lam, rel=1e-15)
    assert not losses["lexlat_k2"].as_error
    assert _rel(_loss_value(losses, "lexlat_k2"), term_oracle) <= 1e-12
    base = sum(_loss_value(losses, k) * s for k, s in (("l_tau", 1.0), ("rate", 3.0), ("agg", 0.1)))
    added = float(rec["total"]) - base
    print(f"T2.5 k2 epoch {epoch}: lam {lam!r} term {term_oracle!r} total - base {added!r}")
    assert abs(added - lam * term_oracle) <= 1e-6 * max(1.0, abs(float(rec["total"])))
