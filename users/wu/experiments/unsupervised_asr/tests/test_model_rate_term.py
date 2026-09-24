"""Oracle tests of the rate term's lattice pieces, ``model/rate_term.py`` (test plan 2026-09-24,
T1.14 and T1.15).

The reference is the brute-force enumerator ``tests/lattice_oracle.py`` on the T1.1 instance
(K = 3, SIL = 2, d 2..3 / SIL 2..5, B = 4, S = [12, 11, 10, 7], T = ceil(S / 3)).
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import lattice_oracle as O

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L
from i6_experiments.users.wu.experiments.unsupervised_asr.model import rate_term as R
from i6_experiments.users.wu.experiments.unsupervised_asr.model.reverse import NEG_INF

K, SIL, D_MIN, D_MAX, D_MAX_SIL, STRIDE = 3, 2, 2, 3, 5, 3
S_LENS = [12, 11, 10, 7]
T_LENS = [4, 4, 4, 3]
N_HIST = {"bigram": K + 1, "trigram": (K + 1) ** 2}
BED_EPS = 0.25  # rate_term.DEFAULT_FD_EPS, the bed's rate_fd_eps


def _cfg(band: int) -> L.LatticeConfig:
    return L.LatticeConfig(n_phones=K, sil_id=SIL, band=band, d_min=D_MIN, d_max=D_MAX,
                           d_max_sil=D_MAX_SIL, topology="blankfree", recognizer_stride=STRIDE)


def _topo(cfg):
    return O.Topology(cfg.n_phones, cfg.sil_id, cfg.band, cfg.d_min, cfg.d_max, cfg.d_max_sil,
                      cfg.recognizer_stride)


def _inputs(history: str, seed: int = 0):
    """The T1.1 generators: log_softmax(2 randn) log_q, randn - 2 seg, log_softmax(1.5 randn) prior."""
    g = torch.Generator().manual_seed(seed)
    b, t_max, s_max = len(S_LENS), max(T_LENS), max(S_LENS)
    log_q = torch.log_softmax(2.0 * torch.randn(b, t_max, K, generator=g, dtype=torch.float64), dim=-1)
    seg = torch.randn(b, K, max(D_MAX, D_MAX_SIL), s_max + 1, generator=g, dtype=torch.float64) - 2.0
    prior = torch.log_softmax(1.5 * torch.randn(N_HIST[history], K, generator=g, dtype=torch.float64), dim=-1)
    return log_q, seg, prior


def _dp_kwargs(cfg, history, prior, tau, beta=1.0, reduction="matmul", checkpoint=3):
    """The ``dp`` dict of ``train_step`` (the same keys, in the same roles)."""
    return dict(
        prior_log_bi=prior, feat_lens=torch.tensor(T_LENS), unit_lens=torch.tensor(S_LENS), cfg=cfg,
        temperature=tau, anchor_weight=0.0, prior_weight=beta, log_q_init=None, collect_stats=True,
        history=L.build_prior_history(cfg, history), reduction=reduction, checkpoint=checkpoint,
    )


def _enums(cfg, history, s_cols):
    return [O.enumerate_utterance(T, S, _topo(cfg), history, s_cols=s_cols) for T, S in zip(T_LENS, S_LENS)]


def _nonsil_mask():
    m = torch.ones(K, dtype=torch.float64)
    m[SIL] = 0.0
    return m


# ---------------------------------------------------------------------------------------------------
# T1.14 expected non-SIL count and the tilt identity
# ---------------------------------------------------------------------------------------------------

T14_GRID = [(w, h, tau) for w in (2, 40) for h in ("bigram", "trigram") for tau in (1.0, 2.0, 8.0)]


@pytest.mark.parametrize("case", T14_GRID, ids=lambda c: "W%d-%s-tau%g" % c)
def test_t1_14_expected_nonsil_and_tilt_identity(case):
    band, history, tau = case
    cfg = _cfg(band)
    log_q, seg, prior = _inputs(history)
    kw = _dp_kwargs(cfg, history, prior, tau)
    out = L.lattice_forward_backward(log_q, seg, **kw)
    got = R.expected_nonsil_tokens(out, cfg)
    mask = _nonsil_mask()
    for b, e in enumerate(_enums(cfg, history, seg.shape[-1])):
        st = O.statistics(e, log_q[b, :T_LENS[b]], seg[b], prior, tau=tau, beta=1.0, n_symbols=K)
        # E[N] = sum pi #{k_u != SIL}
        assert abs(float(got[b]) - float(st["expected_nonsil"])) <= 1e-10, (b, float(got[b]), float(st["expected_nonsil"]))
        # d log Z / db = E[N] / tau, b added to every non-SIL seg entry (oracle autograd in b)
        bt = torch.zeros((), dtype=torch.float64, requires_grad=True)
        tilted = seg[b] + bt * mask.view(-1, 1, 1)
        O.log_z(e, log_q[b, :T_LENS[b]], tilted, prior, tau=tau, beta=1.0).backward()
        assert abs(float(bt.grad) - float(got[b]) / tau) <= 1e-10
        # not expected_tokens: that one counts SIL too
        assert float(out.expected_tokens[b]) >= float(got[b]) - 1e-12


def test_t1_14_tilt_segment_table_shifts_only_non_sil():
    cfg = _cfg(2)
    _, seg, _ = _inputs("trigram")
    seg_in = seg.clone()
    assert R.tilt_segment_table(seg, 0.0, cfg) is seg
    for b in (0.25, -0.25, 3.0):
        t = R.tilt_segment_table(seg, b, cfg)
        assert torch.equal(seg, seg_in)  # out of place
        assert torch.equal(t[:, SIL], seg[:, SIL])  # SIL bitwise untouched
        non = [k for k in range(K) if k != SIL]
        assert torch.equal(t[:, non], seg[:, non] + b)


@pytest.mark.parametrize("b", [0.25, -0.25, 1.0e3])
def test_t1_14_illegal_cells_stay_masked_after_tilt(b):
    """Illegal (k, d, s) -- d outside [d_min, D_k] or s + d > S_b -- stay <= NEG_INF / 2 in the padded
    table the DP reads, and every legal cell is the tilted value / tau."""
    cfg, tau = _cfg(2), 2.0
    _, seg, _ = _inputs("trigram")
    s_cols = seg.shape[-1]
    legal = torch.zeros(seg.shape, dtype=torch.bool)
    for i, S in enumerate(S_LENS):
        for k in range(K):
            dk = D_MAX_SIL if k == SIL else D_MAX
            for d in range(D_MIN, dk + 1):
                for s in range(s_cols):
                    legal[i, k, d - 1, s] = s + d <= S
    raw = seg.masked_fill(~legal, NEG_INF)  # reverse's convention: illegal entries are NEG_INF
    tilted = R.tilt_segment_table(raw, b, cfg)
    pad = L.scaled_seg_pad(tilted, torch.tensor(S_LENS), cfg, tau)
    w = cfg.band
    inner = pad[:, w:w + s_cols, :cfg.d_cap].permute(0, 3, 2, 1)  # back to [B, K, D, S+1]
    assert bool((inner[~legal] <= NEG_INF / 2).all())
    assert torch.allclose(inner[legal], (tilted / tau)[legal], rtol=0, atol=1e-12)
    outside = torch.ones(pad.shape, dtype=torch.bool)
    outside[:, w:w + s_cols, :cfg.d_cap] = False
    assert bool((pad[outside] <= NEG_INF / 2).all())


# ---------------------------------------------------------------------------------------------------
# T1.15 FD surrogate vs the exact derivative
# ---------------------------------------------------------------------------------------------------


def _exact_d_en_d_logq(cfg, history, log_q, seg, prior, tau, beta=1.0):
    """``[B, T, K]`` exact d E[N] / d log_q by autograd of sum softmax(terms) N, and ``[B]`` E[N]."""
    lq = log_q.clone().requires_grad_(True)
    ens = []
    total = 0.0
    for b, e in enumerate(_enums(cfg, history, seg.shape[-1])):
        t = O.terms(e, lq[b, :T_LENS[b]], seg[b], prior, tau=tau, beta=beta)
        en = (torch.softmax(t, dim=0) * e.n_nonsil).sum()
        ens.append(float(en))
        total = total + en
    total.backward()
    return lq.grad, torch.tensor(ens, dtype=torch.float64)


def _norm_rel(g, exact):
    """Per utterance ``max |g - exact| / max |exact|`` over its real frames, then the max over utterances."""
    worst = 0.0
    for b, T in enumerate(T_LENS):
        worst = max(worst, float((g[b, :T] - exact[b, :T]).abs().max() / exact[b, :T].abs().max()))
    return worst


def _fd(cfg, history, log_q, seg, prior, tau, eps, batched):
    kw = _dp_kwargs(cfg, history, prior, tau)
    res = R._fd_passes(log_q, seg, kw, cfg, eps=eps, mode="central", batched=batched)
    g = (res.post_plus - res.post_minus) / (2 * eps)
    fd_expected = tau * (res.log_z[0] - res.log_z[1]) / (2 * eps)
    return res, g, fd_expected


T15_CASES = [(2, "trigram", 2.0), (2, "trigram", 8.0), (40, "bigram", 2.0)]


@pytest.mark.parametrize("case", T15_CASES, ids=lambda c: "W%d-%s-tau%g" % c)
def test_t1_15_fd_surrogate_small_eps(case):
    band, history, tau = case
    cfg = _cfg(band)
    log_q, seg, prior = _inputs(history)
    exact, en = _exact_d_en_d_logq(cfg, history, log_q, seg, prior, tau)
    res_b, g_b, fd_b = _fd(cfg, history, log_q, seg, prior, tau, 1e-4, batched=True)
    res_s, g_s, fd_s = _fd(cfg, history, log_q, seg, prior, tau, 1e-4, batched=False)
    assert res_b.batched and res_b.n_dp_calls == 1 and not res_s.batched and res_s.n_dp_calls == 2
    # stacked = sequential
    assert torch.allclose(res_b.post_plus, res_s.post_plus, rtol=0, atol=1e-12)
    assert torch.allclose(res_b.post_minus, res_s.post_minus, rtol=0, atol=1e-12)
    assert torch.allclose(res_b.log_z, res_s.log_z, rtol=0, atol=1e-12)
    # g = exact d E[N] / d log_q at eps 1e-4 (norm-relative 1e-6), padded frames exactly 0
    rel = _norm_rel(g_b, exact)
    print(f"T1.15 {case} eps 1e-4: max norm-relative |g - exact| = {rel:.3e}")
    assert rel <= 1e-6
    for b, T in enumerate(T_LENS):
        assert float(g_b[b, T:].abs().sum()) == 0.0
    # fd_expected = tau (log Z(+eps) - log Z(-eps)) / 2 eps = E[N]
    rel_en = float(((fd_b - en).abs() / en.abs()).max())
    print(f"T1.15 {case} eps 1e-4: E[N] {en.tolist()} max relative |fd_expected - E[N]| = {rel_en:.3e}")
    assert rel_en <= 1e-7


@pytest.mark.parametrize("case", T15_CASES, ids=lambda c: "W%d-%s-tau%g" % c)
def test_t1_15_fd_bias_at_bed_eps(case):
    """Record the bias of the bed's eps = 0.25 difference; assert the plan's provisional 5e-2."""
    band, history, tau = case
    cfg = _cfg(band)
    log_q, seg, prior = _inputs(history)
    exact, en = _exact_d_en_d_logq(cfg, history, log_q, seg, prior, tau)
    _, g, fd = _fd(cfg, history, log_q, seg, prior, tau, BED_EPS, batched=True)
    rel = _norm_rel(g, exact)
    rel_en = float(((fd - en).abs() / en.abs()).max())
    print(f"T1.15 {case} eps {BED_EPS}: max norm-relative bias of g = {rel:.4e}; "
          f"max relative bias of fd_expected vs E[N] = {rel_en:.4e}")
    assert np.isfinite(rel) and np.isfinite(rel_en)
    assert rel <= 5e-2


def test_t1_15_forward_mode_refused(tmp_path):
    from i6_experiments.users.wu.experiments.unsupervised_asr.model import blankfree_model as BM
    from i6_experiments.users.wu.experiments.unsupervised_asr.model.prior import PhoneNgramPrior

    rng = np.random.RandomState(0)
    n = 40
    prior = PhoneNgramPrior(np.log(np.ones(n) / n), np.log(rng.dirichlet(np.ones(n), size=n + 1)),
                            np.log(rng.dirichlet(np.ones(n), size=(n + 1) * (n + 1))))
    prior.save(str(tmp_path / "prior.npz"))
    np.savez_compressed(str(tmp_path / "eta.npz"), tags=np.array(["u0"]),
                        eta=rng.randn(1, 4).astype(np.float32))
    kw = dict(temperature_schedule=[8.0, 2.0], anchor_weight_schedule=0.0, lam_agg=0.1,
              count_ema_decay=0.99, band=4, prior_weight=1.0, prior_npz_path=str(tmp_path / "prior.npz"),
              eta_table_path=str(tmp_path / "eta.npz"),
              reverse_kwargs={"n_units": 20, "d_max": 4, "d_max_sil": 6, "eta_dim": 4, "d_model": 16,
                              "d_ff": 16},
              lam_rate=3.0, rate_rho_hz=9.6619373279, rate_fd_eps=0.25,
              lattice_reduction="matmul", lattice_checkpoint=2, lattice_float64=True)
    with pytest.raises(ValueError, match="rate_fd_mode"):
        BM.SaeBlankfreeModelV1(rate_fd_mode="forward", **kw)
    assert BM.SaeBlankfreeModelV1(rate_fd_mode="central", **kw).rate_fd_mode == "central"
