"""Oracle tests of the blank-free lattice term ``model/lattice.py`` (test plan 2026-09-24, T1.1-T1.8).

The reference is ``tests/lattice_oracle.py``: an explicit enumeration of every (frame path,
segmentation) latent, independent of the DP. Tolerances are the plan's (section 0): float64 1e-10
relative on log Z and 1e-9 absolute on posteriors / gradients, float32 2e-5 relative on log Z.
"""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

import lattice_oracle as O

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L
from i6_experiments.users.wu.experiments.unsupervised_asr.model.reverse import NEG_INF

# ---------------------------------------------------------------------------------------------------
# the T1.1 instance
# ---------------------------------------------------------------------------------------------------

K, SIL, D_MIN, D_MAX, D_MAX_SIL, STRIDE = 3, 2, 2, 3, 5, 3
S_LENS = [12, 11, 10, 7]
T_LENS = [4, 4, 4, 3]  # = ceil(S / 3)
N_HIST = {"bigram": K + 1, "trigram": (K + 1) ** 2}


def _cfg(band: int, **over) -> L.LatticeConfig:
    kw = dict(n_phones=K, sil_id=SIL, band=band, d_min=D_MIN, d_max=D_MAX, d_max_sil=D_MAX_SIL,
              topology="blankfree", recognizer_stride=STRIDE)
    kw.update(over)
    return L.LatticeConfig(**kw)


def _topo(cfg: L.LatticeConfig) -> O.Topology:
    return O.Topology(cfg.n_phones, cfg.sil_id, cfg.band, cfg.d_min, cfg.d_max, cfg.d_max_sil,
                      cfg.recognizer_stride)


@functools.lru_cache(maxsize=None)
def _inputs(history: str, seed: int = 0):
    """float64 ``(log_q [4,4,3], seg [4,3,5,13], prior [|h|,3])`` of the plan's T1.1 instance."""
    g = torch.Generator().manual_seed(seed)
    b, t_max, s_max = len(S_LENS), max(T_LENS), max(S_LENS)
    log_q = torch.log_softmax(2.0 * torch.randn(b, t_max, K, generator=g, dtype=torch.float64), dim=-1)
    seg = torch.randn(b, K, max(D_MAX, D_MAX_SIL), s_max + 1, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(1.5 * torch.randn(N_HIST[history], K, generator=g, dtype=torch.float64), dim=-1)
    return log_q, seg, prior


@functools.lru_cache(maxsize=None)
def _enum(T: int, S: int, band: int, history: str, s_cols: int, **kw) -> O.Enumeration:
    cfg = _cfg(band, **kw)
    return O.enumerate_utterance(T, S, _topo(cfg), history, s_cols=s_cols)


def _legal(cfg: L.LatticeConfig, s_lens, s_cols: int) -> torch.Tensor:
    """``[B, K, d_cap, s_cols]`` bool, computed here: d in [d_min, D_k] and s + d <= S_b."""
    out = torch.zeros(len(s_lens), cfg.n_phones, cfg.d_cap, s_cols, dtype=torch.bool)
    for b, S in enumerate(s_lens):
        for k in range(cfg.n_phones):
            dk = cfg.d_max_sil if k == cfg.sil_id else cfg.d_max
            for d in range(cfg.d_min, dk + 1):
                for s in range(s_cols):
                    if s + d <= S:
                        out[b, k, d - 1, s] = True
    return out


def _oracle_batch(cfg, history, log_q, seg, prior, tau, beta, s_lens=S_LENS, t_lens=T_LENS):
    """Per-utterance oracle stats (None when infeasible) and the oracle gradients of sum log Z."""
    s_cols = seg.shape[-1]
    lq = log_q.detach().to(torch.float64).clone().requires_grad_(True)
    sg = seg.detach().to(torch.float64).clone().requires_grad_(True)
    pr = prior.detach().to(torch.float64)
    stats, total = [], None
    for b, (T, S) in enumerate(zip(t_lens, s_lens)):
        e = _enum(T, S, cfg.band, history, s_cols, d_max=cfg.d_max, d_max_sil=cfg.d_max_sil)
        st = O.statistics(e, lq[b, :T], sg[b], pr, tau=tau, beta=beta, n_symbols=cfg.n_phones)
        stats.append(st)
        if st is not None:
            lz = O.log_z(e, lq[b, :T], sg[b], pr, tau=tau, beta=beta)
            total = lz if total is None else total + lz
    if total is not None:
        total.backward()
    return stats, (lq.grad if lq.grad is not None else torch.zeros_like(lq)), (
        sg.grad if sg.grad is not None else torch.zeros_like(sg))


_CASES = {}


def _run_case(band, history, tau, beta=1.0, reduction="matmul", checkpoint=3, dtype=torch.float64):
    key = (band, history, tau, beta, reduction, checkpoint, dtype)
    if key in _CASES:
        return _CASES[key]
    cfg = _cfg(band)
    hist = L.build_prior_history(cfg, history)
    log_q64, seg64, prior64 = _inputs(history)
    log_q = log_q64.to(dtype).clone().requires_grad_(True)
    seg = seg64.to(dtype).clone().requires_grad_(True)
    prior = prior64.to(dtype)
    feat_lens, unit_lens = torch.tensor(T_LENS), torch.tensor(S_LENS)
    loss, out = L.lattice_loss(log_q, seg, prior, feat_lens, unit_lens, cfg, temperature=tau,
                               prior_weight=beta, history=hist, reduction=reduction,
                               checkpoint=checkpoint)
    loss.sum().backward()
    stats, g_q, g_seg = _oracle_batch(cfg, history, log_q, seg, prior, tau, beta)
    res = dict(cfg=cfg, loss=loss.detach(), out=out, grad_q=log_q.grad, grad_seg=seg.grad,
               stats=stats, oracle_grad_q=g_q, oracle_grad_seg=g_seg, dtype=dtype)
    _CASES[key] = res
    return res


GRID = [(w, h, tau) for w in (1, 2, 3, 40) for h in ("bigram", "trigram") for tau in (1.0, 2.0, 8.0)]
VARIANTS = (
    [dict(beta=b) for b in (0.0, 0.5)]
    + [dict(reduction=r) for r in ("elementwise", "auto")]
    + [dict(checkpoint=c) for c in (0, 1, 2, 5)]
)
BASE = (2, "trigram", 2.0)


def _ids(case):
    return "W%d-%s-tau%g" % case


# ---------------------------------------------------------------------------------------------------
# T1.1 log Z vs brute force
# ---------------------------------------------------------------------------------------------------


def _check_log_z(res, rtol):
    out, loss = res["out"], res["loss"]
    n_feasible = 0
    for b, st in enumerate(res["stats"]):
        if st is None:
            assert bool(out.z_zero[b]), f"utt {b}: the oracle has no latent but z_zero is False"
            assert float(out.log_z[b]) <= NEG_INF / 2
            assert float(loss[b]) == 0.0
            continue
        n_feasible += 1
        ref = float(st["log_z"])
        assert not bool(out.z_zero[b])
        got = float(out.log_z[b])
        assert abs(got - ref) <= rtol * max(1.0, abs(ref)), (b, got, ref, got - ref)
        assert abs(float(loss[b]) + ref) <= rtol * max(1.0, abs(ref))
    return n_feasible


@pytest.mark.parametrize("case", GRID, ids=_ids)
def test_t1_1_log_z_grid(case):
    band, history, tau = case
    res = _run_case(band, history, tau)
    n = _check_log_z(res, 1e-10)
    # W = 1 excludes S = 10 / T = 4 and S = 7 / T = 3 (|S - 3T| = 2 > W); every other W keeps all four
    assert n == (2 if band == 1 else 4)


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(f"{k}{v[k]}" for k in v))
def test_t1_1_log_z_variants(variant):
    res = _run_case(*BASE, **variant)
    assert _check_log_z(res, 1e-10) == 4


def test_t1_1_log_z_float32():
    res = _run_case(*BASE, dtype=torch.float32)
    assert res["out"].log_z.dtype == torch.float32
    assert _check_log_z(res, 2e-5) == 4


# ---------------------------------------------------------------------------------------------------
# T1.2 gradients, posteriors and statistics vs the oracle
# ---------------------------------------------------------------------------------------------------

ATOL = 1e-9


def _check_t1_2(res):
    cfg, out = res["cfg"], res["out"]
    s_cols = res["grad_seg"].shape[-1]
    # (a) gradients of lattice_loss = autograd of -oracle log Z; illegal seg cells exactly 0
    assert torch.allclose(res["grad_q"], -res["oracle_grad_q"], rtol=0, atol=ATOL), (
        (res["grad_q"] + res["oracle_grad_q"]).abs().max())
    assert torch.allclose(res["grad_seg"], -res["oracle_grad_seg"], rtol=0, atol=ATOL), (
        (res["grad_seg"] + res["oracle_grad_seg"]).abs().max())
    legal = _legal(cfg, S_LENS, s_cols)
    assert bool((res["grad_seg"][~legal] == 0).all())
    for b, st in enumerate(res["stats"]):
        T, S = T_LENS[b], S_LENS[b]
        if st is None:
            assert float(out.post_q[b].abs().sum()) == 0.0 and float(out.seg_post[b].abs().sum()) == 0.0
            continue
        # (b) post_q = oracle; rows sum to 1 before T_b, exactly 0 after
        pq = out.post_q[b]
        assert torch.allclose(pq[:T], st["post_q"], rtol=0, atol=ATOL), (pq[:T] - st["post_q"]).abs().max()
        assert torch.allclose(pq[:T].sum(-1), torch.ones(T, dtype=pq.dtype), rtol=0, atol=ATOL)
        assert bool((pq[T:] == 0).all())
        # (c) seg_post = oracle; sum d * seg_post = S_b; sum seg_post = expected_tokens
        sp = out.seg_post[b]
        assert torch.allclose(sp.reshape(-1), st["seg_post"], rtol=0, atol=ATOL), (
            (sp.reshape(-1) - st["seg_post"]).abs().max())
        d_ax = torch.arange(1, cfg.d_cap + 1, dtype=sp.dtype).view(1, -1, 1)
        assert abs(float((d_ax * sp).sum()) - S) <= ATOL
        assert abs(float(sp.sum()) - float(out.expected_tokens[b])) <= ATOL
        # (d) expected_tokens = E[U]; expected_reverse = E[sum seg] and expected_prior = E[sum P], RAW
        assert abs(float(out.expected_tokens[b]) - float(st["expected_tokens"])) <= ATOL
        assert abs(float(out.expected_reverse[b]) - float(st["expected_reverse"])) <= ATOL
        assert abs(float(out.expected_prior[b]) - float(st["expected_prior"])) <= ATOL


@pytest.mark.parametrize("case", GRID, ids=_ids)
def test_t1_2_gradients_posteriors_stats_grid(case):
    _check_t1_2(_run_case(*case))


@pytest.mark.parametrize("variant", VARIANTS, ids=lambda v: "-".join(f"{k}{v[k]}" for k in v))
def test_t1_2_gradients_posteriors_stats_variants(variant):
    _check_t1_2(_run_case(*BASE, **variant))


def test_t1_2e_finite_difference_on_forward_log_z():
    """(e) central FD (h 1e-6) of forward_log_z at 20 random coordinates = (1/tau) posterior.

    Coordinates are drawn among those whose analytic derivative is >= 0.02 (10 of log_q, 10 of
    seg): at h = 1e-6 the FD's float64 round-off is ~1e-9 absolute, so a 1e-6 relative check is only
    meaningful on derivatives of that size.
    """
    band, history, tau = BASE
    res = _run_case(*BASE)
    cfg, out = res["cfg"], res["out"]
    hist = L.build_prior_history(cfg, history)
    log_q, seg, prior = _inputs(history)
    fl, ul = torch.tensor(T_LENS), torch.tensor(S_LENS)

    def fz(lq, sg):
        with torch.no_grad():
            return L.forward_log_z(lq, sg, prior, fl, ul, cfg, temperature=tau, history=hist,
                                   reduction="matmul")

    an_q = out.post_q / tau
    an_seg = out.seg_post / tau
    g = torch.Generator().manual_seed(1)
    cand_q = (an_q >= 0.02).nonzero()
    cand_s = (an_seg >= 0.02).nonzero()
    pick_q = cand_q[torch.randperm(len(cand_q), generator=g)[:10]]
    pick_s = cand_s[torch.randperm(len(cand_s), generator=g)[:10]]
    assert len(pick_q) == 10 and len(pick_s) == 10
    h = 1e-6
    worst = 0.0
    for idx in pick_q.tolist():
        lp, lm = log_q.clone(), log_q.clone()
        lp[tuple(idx)] += h
        lm[tuple(idx)] -= h
        fd = float((fz(lp, seg)[idx[0]] - fz(lm, seg)[idx[0]]) / (2 * h))
        an = float(an_q[tuple(idx)])
        worst = max(worst, abs(fd - an) / abs(an))
    for idx in pick_s.tolist():
        sp, sm = seg.clone(), seg.clone()
        sp[tuple(idx)] += h
        sm[tuple(idx)] -= h
        fd = float((fz(log_q, sp)[idx[0]] - fz(log_q, sm)[idx[0]]) / (2 * h))
        an = float(an_seg[tuple(idx)])
        worst = max(worst, abs(fd - an) / abs(an))
    print(f"T1.2(e) max relative FD deviation over 20 coordinates: {worst:.3e}")
    assert worst <= 1e-6


# ---------------------------------------------------------------------------------------------------
# T1.3 band convention pin (S2)
# ---------------------------------------------------------------------------------------------------

A, B_ = 0, 1


def _single(cfg, log_q, seg, prior, T, S, tau=1.0, beta=0.0, history="bigram"):
    hist = L.build_prior_history(cfg, history)
    return L.lattice_forward_backward(
        log_q.unsqueeze(0), seg.unsqueeze(0), prior, torch.tensor([T]), torch.tensor([S]), cfg,
        temperature=tau, prior_weight=beta, history=hist, reduction="matmul", checkpoint=0,
    )


def _finite_count(enum, log_q, seg, prior):
    """Latents of ``enum`` whose score is finite (seg cells the instance leaves legal)."""
    if enum.n == 0:
        return 0
    t = O.terms(enum, log_q, seg, prior, tau=1.0, beta=0.0)
    return int((t > NEG_INF / 2).sum())


def test_t1_3_band_convention_instance1():
    """W 2, d_max 4, S 6, T 2, log_q 0: the code admits A(s 0, d 2|3|4) B(s 2|3|4, d 6-s), Z = 3."""
    cfg = _cfg(2, d_max=4, d_max_sil=4)
    T, S = 2, 6
    log_q = torch.zeros(T, K, dtype=torch.float64)
    seg = torch.full((K, cfg.d_cap, S + 1), NEG_INF, dtype=torch.float64)
    for d in (2, 3, 4):
        seg[A, d - 1, 0] = 0.0
    for s in (2, 3, 4):
        seg[B_, 6 - s - 1, s] = 0.0
    prior = torch.zeros(K + 1, K, dtype=torch.float64)
    out = _single(cfg, log_q, seg, prior, T, S)
    z = math.exp(float(out.log_z[0]))
    counts = {m: _finite_count(O.enumerate_utterance(T, S, _topo(cfg), "bigram", band_mode=m),
                               log_q, seg, prior) for m in O.BAND_MODES}
    print(f"T1.3 instance 1: code exp(log Z) = {z!r}; oracle counts by band reading {counts} "
          "(literal = NOTE |s_end - 3 t_emit| <= W)")
    assert abs(z - 3.0) <= 1e-12
    assert counts["code"] == 3
    assert counts["literal"] == 0  # recorded reading of NOTE l.161-163


def test_t1_3_band_convention_instance2_repeat_frame():
    """a = (A, A, B), S 9, T 3, W 2, d_max 6: the repeat frame of A must also stay in band."""
    cfg = _cfg(2, d_max=6, d_max_sil=6)
    T, S = 3, 9
    log_q = torch.full((T, K), NEG_INF, dtype=torch.float64)
    log_q[0, A] = log_q[1, A] = log_q[2, B_] = 0.0
    seg = torch.zeros(K, cfg.d_cap, S + 1, dtype=torch.float64)
    prior = torch.zeros(K + 1, K, dtype=torch.float64)
    out = _single(cfg, log_q, seg, prior, T, S)
    z = math.exp(float(out.log_z[0]))
    counts = {m: O.count(T, S, _topo(cfg), band_mode=m, paths=[(A, A, B_)]) for m in O.BAND_MODES}
    print(f"T1.3 instance 2 (A A B): code exp(log Z) = {z!r}; oracle counts by band reading {counts} "
          "(next_frame = |s_end - 3 (t_emit + 1)| <= W without the repeat frames)")
    assert counts == {"code": 2, "literal": 0, "next_frame": 3}
    assert abs(z - 2.0) <= 1e-12


# ---------------------------------------------------------------------------------------------------
# T1.4 repeat masking and history
# ---------------------------------------------------------------------------------------------------


def test_t1_4a_single_token_string():
    cfg = _cfg(12, d_max=12, d_max_sil=12)
    T, S, tau, beta = 4, 12, 2.0, 1.0
    g = torch.Generator().manual_seed(4)
    log_q = torch.log_softmax(2.0 * torch.randn(T, K, generator=g, dtype=torch.float64), -1)
    seg = torch.full((K, cfg.d_cap, S + 1), NEG_INF, dtype=torch.float64)
    seg[A] = torch.randn(cfg.d_cap, S + 1, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(torch.randn(N_HIST["trigram"], K, generator=g, dtype=torch.float64), -1)
    out = _single(cfg, log_q, seg, prior, T, S, tau=tau, beta=beta, history="trigram")
    start = K * (K + 1) + K  # (BOS, BOS)
    ref = (float(log_q[:, A].sum()) + beta * float(prior[start, A]) + float(seg[A, S - 1, 0])) / tau
    assert abs(float(out.log_z[0]) - ref) <= 1e-12
    assert abs(float(out.expected_tokens[0]) - 1.0) <= 1e-12
    assert abs(float(out.seg_post[0, A, S - 1, 0]) - 1.0) <= 1e-12
    assert abs(float(out.seg_post[0].sum()) - 1.0) <= 1e-12
    assert torch.allclose(out.post_q[0, :, A], torch.ones(T, dtype=torch.float64), rtol=0, atol=1e-12)


def test_t1_4b_history_over_tokens_not_frames():
    cfg = _cfg(40, d_max=5, d_max_sil=5)
    T, S, tau, beta = 3, 9, 2.0, 1.0
    g = torch.Generator().manual_seed(5)
    log_q = torch.full((T, K), NEG_INF, dtype=torch.float64)
    log_q[0, A] = 0.0
    log_q[1, A] = log_q[1, B_] = math.log(0.5)
    log_q[2, B_] = 0.0
    seg = torch.randn(K, cfg.d_cap, S + 1, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(torch.randn(N_HIST["trigram"], K, generator=g, dtype=torch.float64), -1)
    bos = K
    prior[bos * (K + 1) + A, B_] = math.log(0.9)  # P[(BOS, A), B]
    prior[A * (K + 1) + A, B_] = math.log(1e-4)  # P[(A, A), B]: read only by a DP advancing h on repeats
    out = _single(cfg, log_q, seg, prior, T, S, tau=tau, beta=beta, history="trigram")
    enum = O.enumerate_utterance(T, S, _topo(cfg), "trigram", paths=[(A, A, B_), (A, B_, B_)])
    st = O.statistics(enum, log_q, seg, prior, tau=tau, beta=beta, n_symbols=K)
    assert abs(float(out.log_z[0]) - float(st["log_z"])) <= 1e-10 * max(1.0, abs(float(st["log_z"])))
    assert abs(float(out.expected_prior[0]) - float(st["expected_prior"])) <= 1e-10
    hand = float(prior[bos * (K + 1) + bos, A]) + math.log(0.9)
    assert abs(float(out.expected_prior[0]) - hand) <= 1e-10


def _sil_only_log_q(T):
    log_q = torch.full((T, K), NEG_INF, dtype=torch.float64)
    log_q[:, SIL] = 0.0
    return log_q


def test_t1_4c_sil_only_is_one_token():
    """Blank-free: SIL SIL never splits into two tokens (the repeat diagonal covers SIL too).

    D_sil = 12: the only latent is one SIL token of 12 frames.
    """
    T, S = 4, 12
    log_q = _sil_only_log_q(T)
    prior = torch.zeros(K + 1, K, dtype=torch.float64)
    cfg = _cfg(12, d_max=3, d_max_sil=12)
    seg = torch.zeros(K, cfg.d_cap, S + 1, dtype=torch.float64)
    out = _single(cfg, log_q, seg, prior, T, S)
    assert abs(float(out.expected_tokens[0]) - 1.0) <= 1e-12
    assert O.count(T, S, _topo(cfg), paths=[(SIL,) * T]) == 1
    assert abs(float(out.log_z[0])) <= 1e-12  # one latent of score 0


_LOGMM_FLOOR = (
    "FINDING model/lattice.py:673-674 (_logmm): a product whose operands have finite maxima but no "
    "aligned finite pair is floored to a_max + b_max - 708 instead of NEG_INF, so on the matmul path "
    "a Z = 0 utterance comes out with log Z ~ -708, z_zero False and nonzero posteriors"
)


@pytest.mark.parametrize("reduction", [
    "elementwise",
    pytest.param("matmul", marks=pytest.mark.xfail(strict=True, reason=_LOGMM_FLOOR)),
])
def test_t1_4c_sil_sil_is_not_a_latent(reduction):
    """D_sil = 6: two SIL tokens of 6 frames would fit S = 12, but SIL SIL is not a latent -> Z = 0."""
    T, S = 4, 12
    log_q = _sil_only_log_q(T)
    prior = torch.zeros(K + 1, K, dtype=torch.float64)
    cfg = _cfg(12, d_max=3, d_max_sil=6)
    seg = torch.zeros(K, cfg.d_cap, S + 1, dtype=torch.float64)
    hist = L.build_prior_history(cfg, "bigram")
    out = L.lattice_forward_backward(
        log_q.unsqueeze(0), seg.unsqueeze(0), prior, torch.tensor([T]), torch.tensor([S]), cfg,
        temperature=1.0, prior_weight=0.0, history=hist, reduction=reduction)
    assert O.count(T, S, _topo(cfg), paths=[(SIL,) * T]) == 0
    print(f"T1.4(c) SIL SIL, reduction={reduction}: log Z = {float(out.log_z[0])!r}, "
          f"z_zero = {bool(out.z_zero[0])}, expected_tokens = {float(out.expected_tokens[0])!r}")
    assert bool(out.z_zero[0]) and float(out.log_z[0]) <= NEG_INF / 2
    assert float(out.post_q.abs().sum()) == 0.0


# ---------------------------------------------------------------------------------------------------
# T1.5 Z = 0 and padding invariance
# ---------------------------------------------------------------------------------------------------


T15_T, T15_S = [4, 2], [11, 4]


def _t1_5_run(reduction):
    """d_max = d_max_sil = 2, W 6: S 11 / T 4 is infeasible (odd S), S 4 / T 2 the feasible row.

    The plan's feasible row "S = 10" cannot be feasible at d_max = 2 under the train step's
    T = ceil(S / 3) (at most 2T = 8 unit frames); S = 4, T = 2 is the feasible row used instead, and
    W = 6 is this test's choice (the plan leaves W open).
    """
    cfg = _cfg(6, d_max=2, d_max_sil=2)
    tau, beta, history = 2.0, 1.0, "trigram"
    hist = L.build_prior_history(cfg, history)
    g = torch.Generator().manual_seed(6)
    log_q = torch.log_softmax(2.0 * torch.randn(2, 4, K, generator=g, dtype=torch.float64), -1)
    seg = torch.randn(2, K, cfg.d_cap, 12, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(torch.randn(N_HIST[history], K, generator=g, dtype=torch.float64), -1)
    lq, sg = log_q.clone().requires_grad_(True), seg.clone().requires_grad_(True)
    loss, out = L.lattice_loss(lq, sg, prior, torch.tensor(T15_T), torch.tensor(T15_S), cfg,
                               temperature=tau, prior_weight=beta, history=hist, reduction=reduction,
                               checkpoint=2)
    loss.sum().backward()
    return dict(cfg=cfg, tau=tau, beta=beta, history=history, hist=hist, log_q=log_q, seg=seg,
                prior=prior, loss=loss.detach(), out=out, grad_q=lq.grad, grad_seg=sg.grad)


@pytest.mark.parametrize("reduction", [
    "elementwise",
    pytest.param("matmul", marks=pytest.mark.xfail(strict=True, reason=_LOGMM_FLOOR)),
])
def test_t1_5_z_zero_row(reduction):
    r = _t1_5_run(reduction)
    out = r["out"]
    assert O.count(4, 11, _topo(r["cfg"])) == 0
    print(f"T1.5 S=11 T=4, reduction={reduction}: log Z = {float(out.log_z[0])!r}, "
          f"z_zero = {bool(out.z_zero[0])}, loss = {float(r['loss'][0])!r}")
    assert bool(out.z_zero[0]) and float(out.log_z[0]) <= NEG_INF / 2
    assert float(r["loss"][0]) == 0.0
    assert float(r["grad_q"][0].abs().sum()) == 0.0 and float(r["grad_seg"][0].abs().sum()) == 0.0
    for name in ("expected_tokens", "expected_reverse", "expected_prior"):
        assert float(getattr(out, name)[0]) == 0.0, name


@pytest.mark.parametrize("reduction", ["elementwise", "matmul"])
def test_t1_5_feasible_row_padding_invariance(reduction):
    """The feasible row of the batch equals its single run (1e-12) and the oracle (1e-10)."""
    r = _t1_5_run(reduction)
    out, cfg, T, S = r["out"], r["cfg"], T15_T[1], T15_S[1]
    assert not bool(out.z_zero[1])
    single = L.lattice_forward_backward(
        r["log_q"][1:, :T], r["seg"][1:, :, :, :S + 1], r["prior"], torch.tensor([T]), torch.tensor([S]),
        cfg, temperature=r["tau"], prior_weight=r["beta"], history=r["hist"], reduction=reduction,
        checkpoint=2)
    assert abs(float(out.log_z[1]) - float(single.log_z[0])) <= 1e-12
    assert torch.allclose(out.post_q[1, :T], single.post_q[0], rtol=0, atol=1e-12)
    assert float(out.post_q[1, T:].abs().sum()) == 0.0
    assert torch.allclose(out.seg_post[1, :, :, :S + 1], single.seg_post[0], rtol=0, atol=1e-12)
    assert float(out.seg_post[1, :, :, S + 1:].abs().sum()) == 0.0
    for name in ("expected_tokens", "expected_reverse", "expected_prior"):
        assert abs(float(getattr(out, name)[1]) - float(getattr(single, name)[0])) <= 1e-12, name
    enum = O.enumerate_utterance(T, S, _topo(cfg), r["history"], s_cols=12)
    assert enum.n > 0
    ref = float(O.log_z(enum, r["log_q"][1, :T], r["seg"][1], r["prior"], tau=r["tau"], beta=r["beta"]))
    assert abs(float(out.log_z[1]) - ref) <= 1e-10 * max(1.0, abs(ref))


@pytest.mark.parametrize("reduction", ["elementwise", "matmul"])
def test_t1_5_production_z_zero_s1(reduction):
    """The one Z = 0 case the train step can produce at the bed (S = 1, T = 1): detected on both paths."""
    cfg = L.LatticeConfig(n_phones=40, sil_id=39, band=25, d_min=2, d_max=25, d_max_sil=50,
                          topology="blankfree", recognizer_stride=3)
    hist = L.build_prior_history(cfg, "trigram")
    g = torch.Generator().manual_seed(8)
    log_q = torch.log_softmax(torch.randn(2, 2, 40, generator=g, dtype=torch.float64), -1)
    seg = torch.randn(2, 40, 50, 7, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(torch.randn(41 * 41, 40, generator=g, dtype=torch.float64), -1)
    out = L.lattice_forward_backward(log_q, seg, prior, torch.tensor([1, 2]), torch.tensor([1, 6]), cfg,
                                     temperature=2.0, history=hist, reduction=reduction, checkpoint=2)
    assert bool(out.z_zero[0]) and not bool(out.z_zero[1])
    assert float(out.post_q[0].abs().sum()) == 0.0


# ---------------------------------------------------------------------------------------------------
# T1.6 production configuration, through the train step's own arguments
# ---------------------------------------------------------------------------------------------------

# test_lm_phone_prior.CORPUS, copied (that file is edited concurrently; the text is the fixture)
PRIOR_CORPUS = ["<SIL> AA B <SIL>", "<SIL> AA B AA <SIL>", "B IY", "<SIL> IY IY B <SIL>"] * 30


_SIL_SPLIT = (
    "FINDING model/emc_model.py:367 + model/blankfree_model.py:481: model.prior_history is built while "
    "lattice_cfg.topology is still 'ctc' (SIL exempt from the repeat diagonal, lattice.py:355-357) and "
    "is not rebuilt when the blank-free model switches the topology, so the train step's lattice lets a "
    "SIL run split into several SIL tokens -- latents outside the run-collapse definition (NOTE 4.1); "
    "the default (sil_run_collapse=False) keeps it, the option fixes it (test_t1_6_rc_*)"
)


@pytest.fixture(scope="module")
def t1_6(tmp_path_factory):
    """One blank-free train step at the T1.6 configuration, with the l_tau DP call recorded."""
    return _t1_6_step(tmp_path_factory.mktemp("t1_6"))


@pytest.fixture(scope="module")
def t1_6_rc(tmp_path_factory):
    """:func:`t1_6` with the model option ``sil_run_collapse=True`` (arm ``ctrl_20_rc``), nothing else."""
    return _t1_6_step(tmp_path_factory.mktemp("t1_6_rc"), sil_run_collapse=True)


def _t1_6_step(tmp, **model_over):
    """The body of :func:`t1_6`; ``model_over`` are extra model keywords (default: none)."""
    import returnn.frontend as rf
    from returnn.frontend._backend import select_backend_torch
    from returnn.tensor import Dim, Tensor, TensorDict

    from i6_experiments.users.wu.experiments.unsupervised_asr.lm import phone_prior as PP
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import train_step as TS

    select_backend_torch()
    counts, _ = PP.count_ngrams([line.split() for line in PRIOR_CORPUS])
    prior = PP.PhoneNgramPrior.from_counts(counts)
    assert prior.log_tri.shape == (41 * 41, 40)
    prior_path, eta_path = str(tmp / "prior.npz"), str(tmp / "eta.npz")
    prior.save(prior_path)
    tags, eta_dim, n_units = ["u0", "u1"], 4, 20
    rng = np.random.RandomState(0)
    np.savez_compressed(eta_path, tags=np.array(tags), eta=rng.randn(2, eta_dim).astype(np.float32))
    schedule = [8.0, 2.0, 2.0]
    epoch = 2
    torch.manual_seed(0)  # phi / theta init
    model = BM.SaeBlankfreeModelV1(
        temperature_schedule=schedule, anchor_weight_schedule=0.0, lam_agg=0.1,
        count_ema_decay=0.99, band=25, prior_weight=1.0, prior_npz_path=prior_path,
        eta_table_path=eta_path,
        reverse_kwargs={"n_units": n_units, "eta_dim": eta_dim, "d_model": 16, "d_ff": 16},
        lam_rate=3.0, rate_rho_hz=9.6619373279, rate_fd_eps=0.25, rate_fd_mode="central",
        lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True, **model_over,
    )
    s_lens = [6, 5]
    g = torch.Generator().manual_seed(7)
    batch = Dim(2, name="batch")
    tdim = Dim(name="time", dimension=None, dyn_size_ext=Tensor(
        "t", raw_tensor=torch.tensor(s_lens, dtype=torch.int32), dims=[batch], dtype="int32"))
    sdim = Dim(name="units_time", dimension=None, dyn_size_ext=Tensor(
        "s", raw_tensor=torch.tensor(s_lens, dtype=torch.int32), dims=[batch], dtype="int32"))
    feat = Dim(1024, name="feat")
    data = TensorDict()
    data.data["features"] = Tensor("features", dims=[batch, tdim, feat], dtype="float16", feature_dim=feat,
                                   raw_tensor=torch.randn(2, max(s_lens), 1024, generator=g).half())
    units = torch.randint(0, n_units, (2, max(s_lens)), generator=g, dtype=torch.int32)
    data.data["units"] = Tensor("units", dims=[batch, sdim], dtype="int32",
                                sparse_dim=Dim(n_units, name="u"), raw_tensor=units)
    odim = Dim(1, name="o")
    data.data["original_length"] = Tensor("original_length", dims=[batch, odim], dtype="int32",
                                          raw_tensor=torch.tensor([[11], [5]], dtype=torch.int32))
    tag = Tensor("seq_tag", dims=[batch], dtype="string")
    tag.raw_tensor = np.array(tags)
    data.data["seq_tag"] = tag

    captured = []
    real = TS.lattice_loss

    def recording(log_q, seg_table, **kw):
        loss, out = real(log_q, seg_table, **kw)
        captured.append(dict(log_q=log_q.detach().clone(), seg=seg_table.detach().clone(), kw=kw, out=out))
        return loss, out

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(TS, "lattice_loss", recording)
        model.train()
        torch.manual_seed(0)
        rf.init_train_step_run_ctx(train_flag=True, step=0, epoch=epoch)
        TS.train_step(model=model, extern_data=data)
    assert len(captured) == 1
    return dict(model=model, prior=prior, schedule=schedule, epoch=epoch, s_lens=s_lens, units=units,
                tags=tags, **captured[0])


def test_t1_6_train_step_arguments(t1_6):
    model, kw, prior = t1_6["model"], t1_6["kw"], t1_6["prior"]
    cfg = model.lattice_cfg
    assert (cfg.band, cfg.d_min, cfg.d_max, cfg.d_max_sil, cfg.recognizer_stride, cfg.topology) == (
        25, 2, 25, 50, 3, "blankfree")
    assert kw["cfg"] is cfg
    assert kw["temperature"] == t1_6["schedule"][t1_6["epoch"] - 1] == 2.0
    assert kw["prior_weight"] == 1.0
    assert kw["history"] is model.prior_history and model.prior_history.name == "trigram"
    assert kw["prior_log_bi"].dtype == torch.float64
    assert torch.equal(kw["prior_log_bi"], model.prior_log_bi.double())
    assert float((model.prior_log_bi.double() - torch.as_tensor(prior.log_tri)).abs().max()) <= 1e-6
    assert kw["feat_lens"].tolist() == [math.ceil(s / 3) for s in t1_6["s_lens"]] == [2, 2]
    assert kw["unit_lens"].tolist() == t1_6["s_lens"]
    assert (kw["reduction"], kw["checkpoint"]) == ("matmul", 2)
    assert t1_6["log_q"].dtype == torch.float64 and t1_6["log_q"].shape == (2, 2, 40)
    eta = model.lookup_eta(t1_6["tags"], "cpu")
    with torch.no_grad():
        seg_ref = L.build_segment_table(model.reverse, t1_6["units"].long(), eta).double()
    assert torch.equal(t1_6["seg"], seg_ref)


def _t1_6_oracle(t1_6, b, *, sil_split):
    """log Z of utterance b by enumeration. The prior is the fitted log_tri rounded to float32 (the
    model's buffer dtype) and walked here over the TOKENS with (BOS, BOS) padding -- not the package
    scorer (PhoneNgramPrior.per_token_log_probs double-counts a one-token string at order 3,
    model/prior.py:246, pinned in test_lm_phone_prior.py) and not the DP's history map."""
    cfg, kw = t1_6["model"].lattice_cfg, t1_6["kw"]
    tri32 = t1_6["prior"].log_tri.astype(np.float32).astype(np.float64)
    bos, n_ctx = 40, 41

    def walk(tokens):
        h2, h1, total = bos, bos, 0.0
        for k in tokens:
            total += float(tri32[h2 * n_ctx + h1, k])
            h2, h1 = h1, k
        return total

    S = t1_6["s_lens"][b]
    T = math.ceil(S / 3)
    enum = O.enumerate_utterance(T, S, _topo(cfg), "trigram", s_cols=t1_6["seg"].shape[-1], sil_split=sil_split)
    ac, _, rv = O.pieces(enum, t1_6["log_q"][b, :T], t1_6["seg"][b], kw["prior_log_bi"])
    walked = {}
    for _, tokens, _ in enum.latents:
        if tokens not in walked:
            walked[tokens] = walk(tokens)
    pr = torch.tensor([walked[tokens] for _, tokens, _ in enum.latents], dtype=torch.float64)
    return float(torch.logsumexp((ac + 1.0 * pr + rv) / kw["temperature"], dim=0)), enum.n


def _t1_6_compare(t1_6, log_z, *, sil_split, label):
    for b, S in enumerate(t1_6["s_lens"]):
        ref, n = _t1_6_oracle(t1_6, b, sil_split=sil_split)
        got = float(log_z[b])
        print(f"T1.6 {label} utt {b}: S={S} latents={n} log Z code={got:.13f} oracle={ref:.13f} "
              f"diff={got - ref:.3e}")
        assert abs(got - ref) <= 1e-10 * max(1.0, abs(ref)), (b, got, ref)


@pytest.mark.xfail(strict=True, reason=_SIL_SPLIT)
def test_t1_6_log_z_vs_run_collapse_oracle(t1_6):
    """The plan's T1.6 assertion: the captured log Z = the run-collapse oracle (1e-10)."""
    _t1_6_compare(t1_6, t1_6["out"].log_z, sil_split=False, label="step vs run-collapse")


def test_t1_6_log_z_is_the_sil_split_lattice(t1_6):
    """What the step computes: the oracle with SIL runs splittable into several SIL tokens (1e-10)."""
    _t1_6_compare(t1_6, t1_6["out"].log_z, sil_split=True, label="step vs sil-split")


def test_t1_6_blankfree_history_matches_run_collapse_oracle(t1_6):
    """The same captured inputs through the DP with the history rebuilt from the blank-free
    lattice_cfg: equal to the run-collapse oracle (1e-10), so the DP itself is right."""
    kw = dict(t1_6["kw"])
    kw["history"] = L.build_prior_history(kw["cfg"], "trigram")
    out = L.lattice_forward_backward(t1_6["log_q"], t1_6["seg"], **kw)
    _t1_6_compare(t1_6, out.log_z, sil_split=False, label="blank-free history vs run-collapse")


@pytest.mark.xfail(strict=True, reason=_SIL_SPLIT)
def test_t1_6_model_history_is_the_blankfree_one(t1_6):
    model = t1_6["model"]
    rebuilt = L.build_prior_history(model.lattice_cfg, "trigram")
    sil = model.lattice_cfg.sil_id
    h = 40 * 41 + sil  # (BOS, SIL)
    print(f"T1.6 same_nonsil[(BOS, SIL), SIL]: model {bool(model.prior_history.same_nonsil[0, 0, h, sil])}, "
          f"rebuilt from the blank-free cfg {bool(rebuilt.same_nonsil[0, 0, h, sil])}")
    assert torch.equal(model.prior_history.same_nonsil, rebuilt.same_nonsil)


# --- the fix: ``sil_run_collapse=True`` (arm ctrl_20_rc) -------------------------------------------


def test_t1_6_rc_train_step_arguments(t1_6, t1_6_rc):
    """Option on: the step's l_tau call is the default's in every argument but the history, which is
    the model's own, rebuilt from the blank-free cfg; the step's inputs are the default's bit for bit."""
    model, kw = t1_6_rc["model"], t1_6_rc["kw"]
    assert model.sil_run_collapse is True and t1_6["model"].sil_run_collapse is False
    assert kw["history"] is model.prior_history and model.prior_history.name == "trigram"
    rebuilt = L.build_prior_history(model.lattice_cfg, "trigram")
    assert torch.equal(model.prior_history.same_nonsil, rebuilt.same_nonsil)
    sil, h = model.lattice_cfg.sil_id, 40 * 41 + model.lattice_cfg.sil_id  # (BOS, SIL)
    assert bool(model.prior_history.same_nonsil[0, 0, h, sil])  # SIL -> SIL emit from f = 1 forbidden
    assert not bool(t1_6["model"].prior_history.same_nonsil[0, 0, h, sil])  # default: still allowed
    for name in ("succ", "last", "next", "group", "members", "dst"):
        assert torch.equal(getattr(model.prior_history, name), getattr(t1_6["model"].prior_history, name)), name
    # only SIL's diagonal differs from the default history
    diff = model.prior_history.same_nonsil ^ t1_6["model"].prior_history.same_nonsil
    assert diff.any() and bool((diff.nonzero()[:, -1] == sil).all())
    assert model.lattice_cfg == t1_6["model"].lattice_cfg
    assert set(kw) == set(t1_6["kw"])
    for name, v in kw.items():
        if name == "history":
            continue
        ref = t1_6["kw"][name]
        assert (torch.equal(v, ref) if torch.is_tensor(v) else v == ref), name
    assert torch.equal(t1_6_rc["log_q"], t1_6["log_q"]) and torch.equal(t1_6_rc["seg"], t1_6["seg"])


def test_t1_6_rc_log_z_vs_run_collapse_oracle(t1_6_rc):
    """The plan's T1.6 assertion with the option on: the captured log Z = the run-collapse oracle (1e-10)."""
    _t1_6_compare(t1_6_rc, t1_6_rc["out"].log_z, sil_split=False, label="rc step vs run-collapse")


def test_t1_6_rc_log_z_is_not_the_sil_split_lattice(t1_6, t1_6_rc):
    """Option on: the SIL-split oracle no longer matches, and log Z sums over a strict subset of the
    default's latents (G0.RC: rc log Z <= ctrl_20's on the same inputs).  The split latents are few at
    this tiny shape (3 of 4723 and 2 of 3162): utterance 0's log Z moves by ~4e-8, 18x T1.6's tolerance,
    utterance 1's by ~2e-11, inside it -- so the mismatch is required of at least one utterance and the
    strict decrease of every one."""
    mismatched = []
    for b, S in enumerate(t1_6_rc["s_lens"]):
        ref_split, n_split = _t1_6_oracle(t1_6_rc, b, sil_split=True)
        ref_rc, n_rc = _t1_6_oracle(t1_6_rc, b, sil_split=False)
        got, got_default = float(t1_6_rc["out"].log_z[b]), float(t1_6["out"].log_z[b])
        print(f"T1.6 rc utt {b}: S={S} latents rc={n_rc} split={n_split} log Z rc={got:.13f} "
              f"default={got_default:.13f} sil-split oracle={ref_split:.13f} diff={got - ref_split:.3e}")
        assert n_rc < n_split
        assert got < got_default and abs(got_default - ref_split) <= 1e-10 * max(1.0, abs(ref_split))
        mismatched.append(abs(got - ref_split) > 1e-10 * max(1.0, abs(ref_split)))  # T1.6's tolerance
    assert mismatched[0] and any(mismatched), mismatched


# ---------------------------------------------------------------------------------------------------
# T1.7 consistency at production shape (slow) and T1.8 GPU parity
# ---------------------------------------------------------------------------------------------------

P_S = [90, 61, 40]
P_T = [math.ceil(s / 3) for s in P_S]  # [30, 21, 14]
P_K, P_SIL = 40, 39


def _prod_cfg():
    return L.LatticeConfig(n_phones=P_K, sil_id=P_SIL, band=25, d_min=2, d_max=25, d_max_sil=50,
                           topology="blankfree", recognizer_stride=3)


@functools.lru_cache(maxsize=None)
def _prod_inputs():
    """Random tables at the production shape (T1.1's generators: 2 randn log_q, randn - 2 seg,
    1.5 randn prior)."""
    cfg = _prod_cfg()
    g = torch.Generator().manual_seed(17)
    log_q = torch.log_softmax(2.0 * torch.randn(3, max(P_T), P_K, generator=g, dtype=torch.float64), -1)
    seg = torch.randn(3, P_K, cfg.d_cap, max(P_S) + 1, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(1.5 * torch.randn((P_K + 1) ** 2, P_K, generator=g, dtype=torch.float64), -1)
    return log_q, seg, prior


def _prod_fb(log_q, seg, prior, *, reduction="matmul", checkpoint=7, s_lens=P_S, t_lens=P_T, tau=2.0):
    cfg = _prod_cfg()
    hist = L.build_prior_history(cfg, "trigram")
    return L.lattice_forward_backward(
        log_q, seg, prior, torch.tensor(t_lens), torch.tensor(s_lens), cfg, temperature=tau,
        prior_weight=1.0, history=hist, reduction=reduction, checkpoint=checkpoint)


def _prod_fz(log_q, seg, prior, tau=2.0):
    cfg = _prod_cfg()
    hist = L.build_prior_history(cfg, "trigram")
    with torch.no_grad():
        return L.forward_log_z(log_q, seg, prior, torch.tensor(P_T), torch.tensor(P_S), cfg,
                               temperature=tau, prior_weight=1.0, history=hist, reduction="matmul")


def _same(a, b):
    return all(torch.equal(getattr(a, f), getattr(b, f)) for f in a._fields)


def _max_diff(a, b):
    return {f: float((getattr(a, f).double() - getattr(b, f).double()).abs().max()) for f in a._fields
            if f != "z_zero"}


@pytest.mark.slow
def test_t1_7_production_shape_consistency():
    cfg = _prod_cfg()
    tau = 2.0
    log_q, seg, prior = _prod_inputs()
    ref = _prod_fb(log_q, seg, prior, checkpoint=7)
    assert not bool(ref.z_zero.any())
    # (i) elementwise vs matmul
    ew = _prod_fb(log_q, seg, prior, reduction="elementwise", checkpoint=7)
    d = _max_diff(ew, ref)
    print(f"T1.7 (i) elementwise vs matmul max |diff|: {d}")
    assert all(v <= 1e-9 for v in d.values()), d
    # (ii) checkpoint 0 / 7 / 32 bit-identical
    for ck in (0, 32):
        assert _same(_prod_fb(log_q, seg, prior, checkpoint=ck), ref), ck
    # (iii) fp32 vs fp64 log Z
    f32 = _prod_fb(log_q.float(), seg.float(), prior.float(), checkpoint=7)
    rel = ((f32.log_z.double() - ref.log_z).abs() / ref.log_z.abs())
    print(f"T1.7 (iii) fp32 vs fp64 log Z: fp64 {ref.log_z.tolist()} relative diff {rel.tolist()}")
    assert float(rel.max()) <= 1e-5
    # (iv) batched vs single
    for b in range(3):
        one = _prod_fb(log_q[b:b + 1, :P_T[b]], seg[b:b + 1, :, :, :P_S[b] + 1], prior, checkpoint=7,
                       s_lens=[P_S[b]], t_lens=[P_T[b]])
        assert abs(float(one.log_z[0]) - float(ref.log_z[b])) <= 1e-10 * abs(float(ref.log_z[b]))
        assert torch.allclose(one.post_q[0], ref.post_q[b, :P_T[b]], rtol=0, atol=1e-10)
        assert torch.allclose(one.seg_post[0], ref.seg_post[b, :, :, :P_S[b] + 1], rtol=0, atol=1e-10)
    # (v) sums
    nonsil = torch.ones(P_K, dtype=torch.float64)
    nonsil[P_SIL] = 0.0
    d_ax = torch.arange(1, cfg.d_cap + 1, dtype=torch.float64).view(1, -1, 1)
    for b in range(3):
        rows = ref.post_q[b, :P_T[b]].sum(-1)
        assert torch.allclose(rows, torch.ones_like(rows), rtol=0, atol=1e-9)
        assert float(ref.post_q[b, P_T[b]:].abs().sum()) == 0.0
        sp = ref.seg_post[b]
        assert abs(float((d_ax * sp).sum()) - P_S[b]) <= 1e-9 * P_S[b]
        assert abs(float(sp.sum()) - float(ref.expected_tokens[b])) <= 1e-9
        e_nonsil = float((sp.sum(dim=(1, 2)) * nonsil).sum())
        assert e_nonsil <= float(ref.expected_tokens[b]) + 1e-9
    # (vi) directional derivatives = (1/tau) <post, delta>
    g = torch.Generator().manual_seed(18)
    dq = torch.randn(log_q.shape, generator=g, dtype=torch.float64)
    ds = torch.randn(seg.shape, generator=g, dtype=torch.float64)
    h = 1e-5
    fd = (_prod_fz(log_q + h * dq, seg + h * ds, prior) - _prod_fz(log_q - h * dq, seg - h * ds, prior)) / (2 * h)
    an = ((ref.post_q * dq).sum(dim=(1, 2)) + (ref.seg_post * ds).sum(dim=(1, 2, 3))) / tau
    rel = ((fd - an).abs() / an.abs())
    print(f"T1.7 (vi) directional derivative relative error {rel.tolist()}")
    assert float(rel.max()) <= 1e-6
    # (vii) tau (log Z(+eps) - log Z(-eps)) / 2 eps = E[N]
    eps = 1e-4
    tilt = torch.zeros_like(seg)
    tilt[:, :P_SIL] = eps
    tilt[:, P_SIL + 1:] = eps
    e_fd = tau * (_prod_fz(log_q, seg + tilt, prior) - _prod_fz(log_q, seg - tilt, prior)) / (2 * eps)
    e_n = (ref.seg_post.sum(dim=(2, 3)) * nonsil).sum(-1)
    rel = ((e_fd - e_n).abs() / e_n.abs())
    print(f"T1.7 (vii) E[N] {e_n.tolist()} FD relative error {rel.tolist()}")
    assert float(rel.max()) <= 1e-6


@pytest.mark.gpu
def test_t1_8_gpu_parity():
    log_q, seg, prior = _prod_inputs()
    cpu64 = _prod_fb(log_q, seg, prior, checkpoint=7)
    cpu32 = _prod_fb(log_q.float(), seg.float(), prior.float(), checkpoint=7)
    dev = torch.device("cuda")
    gpu64 = _prod_fb(log_q.to(dev), seg.to(dev), prior.to(dev), checkpoint=7)
    d = {k: v for k, v in _max_diff(_to_cpu(gpu64), cpu64).items()}
    assert all(v <= 1e-9 for v in d.values()), d
    prev = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        gpu32 = _prod_fb(log_q.float().to(dev), seg.float().to(dev), prior.float().to(dev), checkpoint=7)
        torch.backends.cuda.matmul.allow_tf32 = True
        gpu32_tf = _prod_fb(log_q.float().to(dev), seg.float().to(dev), prior.float().to(dev), checkpoint=7)
        assert torch.backends.cuda.matmul.allow_tf32 is True  # the guard restores the caller's flag
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    rel = ((gpu32.log_z.double().cpu() - cpu32.log_z.double()).abs() / cpu32.log_z.double().abs())
    assert float(rel.max()) <= 1e-4
    assert _same(_to_cpu(gpu32_tf), _to_cpu(gpu32))


def _to_cpu(out):
    return type(out)(*(t.cpu() for t in out))
