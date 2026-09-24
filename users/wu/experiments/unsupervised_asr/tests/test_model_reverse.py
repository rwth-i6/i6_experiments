"""Oracle tests of ``model/reverse.py`` (the segmental reverse model p_phi) and of its identity with
the lattice (test plan 2026-09-24, T1.9, T1.10, T1.11).

Every oracle here is a naive computation written from the definition, not from the code:

* the position bucket of a within-segment frame r is ``floor(r n / d)``, the duration bucket is
  ``d <= dur_bucket_edge``, and a segment's emission score is the explicit sum over its frames;
* ``log p(d | k)`` is the log-softmax of ``dur_logits[k]`` over the legal durations
  ``[d_min, D_k]`` only;
* ``log p_phi(z | y, eta)`` is the logsumexp over every explicit composition of S into
  ``d_1 + ... + d_U`` with ``d_u`` in ``[d_min, D(k_u)]``.

The emission head's categorical (``emission_log_probs``) is the model's definition of nu_phi and is
read, not re-implemented; T1.9(b) checks that the DP's segment table sums it correctly.
"""

import itertools
import math

import numpy as np
import pytest
import torch

from i6_experiments.users.wu.experiments.unsupervised_asr.model import lattice as L
from i6_experiments.users.wu.experiments.unsupervised_asr.model import reverse as R

# T1.9 instance (test plan section 3)
N_TYPES, N_UNITS, SIL, D_MIN, D_MAX, D_MAX_SIL, ETA_DIM, EDGE = 3, 7, 2, 2, 4, 6, 2, 3


def _cfg(n_pos: int = 3) -> R.ReverseConfig:
    return R.ReverseConfig(
        n_types=N_TYPES, n_units=N_UNITS, sil_id=SIL, d_min=D_MIN, d_max=D_MAX, d_max_sil=D_MAX_SIL,
        eta_dim=ETA_DIM, d_model=8, d_ff=8, dur_bucket_edge=EDGE, n_pos_buckets=n_pos,
    )


def _model(n_pos: int = 3, seed: int = 0) -> R.SegmentalReverseModel:
    """A reverse model with every parameter (duration logits included) moved off its init, so the
    emission and the duration tables are far from uniform."""
    torch.manual_seed(seed)
    model = R.SegmentalReverseModel(_cfg(n_pos))
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p))
    return model


# --- independent definitions ----------------------------------------------------------------------


def _d_max(k: int) -> int:
    return D_MAX_SIL if k == SIL else D_MAX


def _dur_bucket(d: int) -> int:
    return 0 if d <= EDGE else 1


def _pos_bucket(r: int, d: int, n: int) -> int:
    return (r * n) // d


def _log_dur(model) -> torch.Tensor:
    """``[n_types, d_cap]`` float64 log p(d | k): log-softmax over [d_min, D_k] only, -inf elsewhere."""
    logits = model.dur_logits.double()
    rows = []
    for k in range(N_TYPES):
        legal = list(range(D_MIN, _d_max(k) + 1))
        lse = torch.logsumexp(torch.stack([logits[k, d - 1] for d in legal]), dim=0)
        rows.append(torch.stack([
            logits[k, d - 1] - lse if d in legal else logits.new_tensor(-math.inf)
            for d in range(1, logits.shape[1] + 1)
        ]))
    return torch.stack(rows)


def _compositions(n_frames: int, y):
    """Every (d_1, ..., d_U) with d_u in [d_min, D(y_u)] and sum n_frames."""
    if not y:
        if n_frames == 0:
            yield ()
        return
    for d in range(D_MIN, _d_max(y[0]) + 1):
        if d <= n_frames:
            for rest in _compositions(n_frames - d, y[1:]):
                yield (d,) + rest


def _oracle_logp_float(emis: np.ndarray, log_dur: np.ndarray, z, y, n_pos: int):
    """float64 ``log p_phi(z | y)`` by explicit enumeration; ``None`` if no segmentation exists.

    :param emis: ``[n_types, n_dur_buckets, n_pos, K]`` log nu for ONE utterance's eta.
    """
    terms = []
    for comp in _compositions(len(z), tuple(y)):
        s, tot = 0, 0.0
        for k, d in zip(y, comp):
            tot += log_dur[k, d - 1]
            for r in range(d):
                tot += emis[k, _dur_bucket(d), _pos_bucket(r, d, n_pos), z[s + r]]
            s += d
        terms.append(tot)
    if not terms:
        return None
    m = max(terms)
    return m + math.log(sum(math.exp(t - m) for t in terms))


def _oracle_logp_torch(model, z, y, eta_row):
    """The same enumeration as a differentiable float64 torch expression (for the gradient check)."""
    n_pos = model.cfg.n_pos_buckets
    emis = model.emission_log_probs(eta_row.view(1, -1))[0].double()
    log_dur = _log_dur(model)
    terms = []
    for comp in _compositions(len(z), tuple(y)):
        s, parts = 0, []
        for k, d in zip(y, comp):
            parts.append(log_dur[k, d - 1])
            for r in range(d):
                parts.append(emis[k, _dur_bucket(d), _pos_bucket(r, d, n_pos), z[s + r]])
            s += d
        terms.append(torch.stack(parts).sum())
    return torch.logsumexp(torch.stack(terms), dim=0)


def _single(model, z, y, eta):
    batch = R.pack_batch([(z, y, eta)], model.cfg)
    return model.forward_logsum(batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"])[0]


# --- T1.9 reverse model tables ----------------------------------------------------------------------


def test_position_bounds_partition_and_closed_form():
    """T1.9(a): bounds partition range(d) in order; r lies in bucket floor(r n / d)."""
    for d in range(1, 61):
        for n in range(1, 6):
            bounds = R.position_bounds(d, n)
            assert len(bounds) == n
            covered = [r for lo, hi in bounds for r in range(lo, hi)]
            assert covered == list(range(d)), (d, n, bounds)
            for j, (lo, hi) in enumerate(bounds):
                for r in range(lo, hi):
                    assert j == _pos_bucket(r, d, n), (d, n, r, j)


@pytest.mark.parametrize("n_pos", [1, 3, 4])
def test_segment_scores_explicit_sum(n_pos):
    """T1.9(b): seg[b, k, d-1, s] = sum_{r<d} emis[b, k, dur_bucket(d), floor(r n/d), z[s+r]]."""
    model = _model(n_pos, seed=n_pos)
    g = torch.Generator().manual_seed(10 + n_pos)
    z = torch.randint(N_UNITS, (2, 13), generator=g)
    eta = torch.randn(2, ETA_DIM, generator=g)
    with torch.no_grad():
        seg = model.segment_scores(z, eta).double().numpy()
        emis = model.emission_log_probs(eta).double().numpy()
    assert seg.shape == (2, N_TYPES, D_MAX_SIL, 14)
    zz = z.numpy()
    n_checked = 0
    for b in range(2):
        for k in range(N_TYPES):
            for d in range(D_MIN, D_MAX_SIL + 1):
                for s in range(0, 13 - d + 1):
                    expect = sum(emis[b, k, _dur_bucket(d), _pos_bucket(r, d, n_pos), zz[b, s + r]]
                                 for r in range(d))
                    assert abs(seg[b, k, d - 1, s] - expect) <= 1e-5 * max(1.0, abs(expect)), (
                        b, k, d, s, seg[b, k, d - 1, s], expect)
                    n_checked += 1
    assert n_checked == 2 * N_TYPES * sum(13 - d + 1 for d in range(D_MIN, D_MAX_SIL + 1))


def test_duration_log_probs_support_and_values():
    """T1.9(c): exp(log p(d|k)) sums to 1 on [d_min, D_k] and is exactly 0 outside."""
    model = _model(3, seed=5)
    with torch.no_grad():
        lp = model.duration_log_probs().double()
        oracle = _log_dur(model)
    p = lp.exp()
    for k in range(N_TYPES):
        legal = list(range(D_MIN, _d_max(k) + 1))
        assert abs(float(p[k, [d - 1 for d in legal]].sum()) - 1.0) <= 1e-6
        for d in range(1, D_MAX_SIL + 1):
            if d in legal:
                assert abs(float(lp[k, d - 1] - oracle[k, d - 1])) <= 1e-5
            else:
                assert float(p[k, d - 1]) == 0.0, (k, d)


def test_build_segment_table_is_scores_plus_durations():
    """T1.9(d): build_segment_table = segment_scores + duration_log_probs (broadcast over b, s)."""
    model = _model(3, seed=6)
    g = torch.Generator().manual_seed(6)
    z = torch.randint(N_UNITS, (2, 13), generator=g)
    eta = torch.randn(2, ETA_DIM, generator=g)
    table = L.build_segment_table(model, z, eta)
    assert table.requires_grad
    with torch.no_grad():
        expect = model.segment_scores(z, eta) + model.duration_log_probs()[None, :, :, None]
    assert torch.equal(table.detach(), expect)
    assert not L.build_segment_table(model, z, eta, detach=True).requires_grad


# --- T1.10 forward_logsum vs enumeration ----------------------------------------------------------


def _all_cases(seed=0):
    """Every y in {0,1,2}^U for U = 1..4 and every S = 2..12, each with its own random z and eta."""
    g = torch.Generator().manual_seed(seed)
    cases = []
    for u in range(1, 5):
        for y in itertools.product(range(N_TYPES), repeat=u):
            for s in range(2, 13):
                z = torch.randint(N_UNITS, (s,), generator=g).tolist()
                eta = torch.randn(ETA_DIM, generator=g).tolist()
                cases.append((z, list(y), eta))
    return cases


@pytest.mark.parametrize("n_pos", [1, 3])
def test_forward_logsum_matches_enumeration(n_pos):
    """T1.10: values (1e-4 rel), feasibility, and the infeasible floor, on every (y, S) case."""
    model = _model(n_pos, seed=20 + n_pos)
    cfg = model.cfg
    cases = _all_cases(seed=n_pos)
    with torch.no_grad():
        log_dur = _log_dur(model).numpy()
        n_feasible = 0
        for lo in range(0, len(cases), 64):
            chunk = cases[lo : lo + 64]
            batch = R.pack_batch(chunk, cfg)
            logp = model.forward_logsum(batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"])
            emis = model.emission_log_probs(batch["eta"]).double().numpy()
            for i, (z, y, _) in enumerate(chunk):
                oracle = _oracle_logp_float(emis[i], log_dur, z, y, n_pos)
                assert R.feasible(len(z), y, cfg) == (oracle is not None), (z, y)
                if oracle is None:
                    assert float(logp[i]) <= R.NEG_INF / 2, (z, y, float(logp[i]))
                else:
                    n_feasible += 1
                    assert abs(float(logp[i]) - oracle) <= 1e-4 * abs(oracle), (z, y, float(logp[i]), oracle)
    assert n_feasible > 300  # the grid is not vacuous


def _mixed_items():
    """A padded batch of feasible items with different S and U, repeats and SIL runs included."""
    g = torch.Generator().manual_seed(77)
    specs = [([0], 3), ([1, 1], 7), ([2, 2, 2], 13), ([0, 2, 1], 9), ([2], 6), ([1, 0, 1, 0], 8),
             ([0, 0], 4), ([2, 1, 2, 2], 17)]
    items = []
    for y, s in specs:
        assert R.feasible(s, y, _cfg())
        items.append((torch.randint(N_UNITS, (s,), generator=g).tolist(), y,
                      torch.randn(ETA_DIM, generator=g).tolist()))
    return items


def test_forward_logsum_padded_batch_equals_single_runs():
    """T1.10: a padded mixed batch gives each row its own single-utterance value."""
    model = _model(3, seed=31)
    items = _mixed_items()
    with torch.no_grad():
        batch = R.pack_batch(items, model.cfg)
        logp = model.forward_logsum(batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"])
        for i, (z, y, eta) in enumerate(items):
            single = float(_single(model, z, y, eta))
            assert abs(float(logp[i]) - single) <= 1e-4 * abs(single), (i, float(logp[i]), single)


def test_forward_logsum_gradients_match_oracle_autograd():
    """T1.10: gradients on every parameter of sum_b log p = those of the enumeration (1e-4 rel,
    read as max |g - g_oracle| <= 1e-4 * max |g_oracle| per parameter tensor)."""
    model = _model(3, seed=41)
    items = _mixed_items()
    batch = R.pack_batch(items, model.cfg)
    model.zero_grad(set_to_none=True)
    model.forward_logsum(batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"]).sum().backward()
    code = {n: p.grad.detach().double().clone() for n, p in model.named_parameters()}
    model.zero_grad(set_to_none=True)
    total = sum(_oracle_logp_torch(model, z, y, torch.tensor(eta)) for z, y, eta in items)
    total.backward()
    for n, p in model.named_parameters():
        oracle = p.grad.detach().double()
        scale = float(oracle.abs().max())
        err = float((code[n] - oracle).abs().max())
        assert err <= 1e-4 * max(scale, 1e-12), (n, err, scale)
    assert float(code["dur_logits"].abs().max()) > 0 and float(code["lin2.weight"].abs().max()) > 0


def test_infeasible_raises_in_sufficient_stats_and_evaluate_rows():
    """T1.10: an infeasible (S, y) makes assert_finite_sufficient_stats raise; a feasible batch does
    not; evaluate()'s per-utterance rows equal forward_logsum."""
    model = _model(3, seed=51)
    cfg = model.cfg
    items = _mixed_items()
    bad = ([0, 1, 2, 3 % N_UNITS, 4, 5, 6, 0, 1, 2], [0], [0.1, -0.2])  # S = 10 > D_0 = 4
    assert not R.feasible(len(bad[0]), bad[1], cfg)
    with torch.no_grad():
        batch = R.pack_batch(items + [bad], cfg)
        logp, stats = model.forward_logsum(
            batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"], return_stats=True)
        assert float(logp[-1]) <= R.NEG_INF / 2
        with pytest.raises(FloatingPointError):
            R.assert_finite_sufficient_stats(stats)
        batch = R.pack_batch(items, cfg)
        _, stats = model.forward_logsum(
            batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"], return_stats=True)
        R.assert_finite_sufficient_stats(stats)
        total, frames, rows = R.evaluate(model, items, batch_size=3, per_utterance=True)
        assert sorted(r["index"] for r in rows) == list(range(len(items)))
        assert frames == sum(len(z) for z, _, _ in items)
        for r in rows:
            z, y, eta = items[r["index"]]
            single = float(_single(model, z, y, eta))
            assert abs(r["log_p"] - single) <= 1e-4 * abs(single), (r, single)
            assert r["frames"] == len(z)
        assert abs(total - sum(r["log_p"] for r in rows)) <= 1e-9 * abs(total)


# --- T1.11 lattice <-> reverse <-> prior identity (S1) ---------------------------------------------


def _runs(path):
    return [k for i, k in enumerate(path) if i == 0 or path[i - 1] != k]


def _identity_inputs():
    model = _model(3, seed=61)
    g = torch.Generator().manual_seed(61)
    t_len, s_len, n_ctx = 4, 10, N_TYPES + 1
    log_q = torch.log_softmax(2.0 * torch.randn(1, t_len, N_TYPES, generator=g, dtype=torch.float64), -1)
    prior = torch.log_softmax(1.5 * torch.randn(n_ctx * n_ctx, N_TYPES, generator=g, dtype=torch.float64), -1)
    z = torch.randint(N_UNITS, (1, s_len), generator=g)
    eta = torch.randn(1, ETA_DIM, generator=g)
    return model, log_q, prior, z, eta


def _note_terms(model, log_q, prior, z, eta):
    """Per path a in {0,1,2}^T: sum_t log q + sum_u P[h_u, k_u] (raw trigram, BOS padding) and
    reverse.forward_logsum(z, B(a), eta). Infeasible strings are dropped (their p_phi is 0)."""
    t_len = log_q.shape[1]
    bos, n_ctx = N_TYPES, N_TYPES + 1
    paths = list(itertools.product(range(N_TYPES), repeat=t_len))
    strings = [_runs(a) for a in paths]
    with torch.no_grad():
        items = [(z[0].tolist(), y, eta[0].tolist()) for y in strings]
        batch = R.pack_batch(items, model.cfg)
        rev = model.forward_logsum(batch["z"], batch["y"], batch["eta"], batch["s_lens"], batch["u_lens"])
    q_terms, rev_terms = [], []
    for i, (a, y) in enumerate(zip(paths, strings)):
        if not R.feasible(z.shape[1], y, model.cfg):
            assert float(rev[i]) <= R.NEG_INF / 2
            continue
        lq = sum(float(log_q[0, t, a[t]]) for t in range(t_len))
        h2, h1, lp = bos, bos, 0.0
        for k in y:
            lp += float(prior[h2 * n_ctx + h1, k])
            h2, h1 = h1, k
        q_terms.append(lq + lp)
        rev_terms.append(float(rev[i]))
    return np.array(q_terms), np.array(rev_terms)


def _lattice_log_z(model, log_q, prior, z, eta, tau):
    cfg = L.LatticeConfig(n_phones=N_TYPES, sil_id=SIL, band=60, d_min=D_MIN, d_max=D_MAX,
                          d_max_sil=D_MAX_SIL, topology="blankfree", recognizer_stride=3)
    hist = L.build_prior_history(cfg, "trigram")
    with torch.no_grad():
        seg = L.build_segment_table(model, z, eta).double()
    out = L.lattice_forward_backward(
        log_q, seg, prior, torch.tensor([log_q.shape[1]]), torch.tensor([z.shape[1]]), cfg,
        temperature=tau, prior_weight=1.0, history=hist,
    )
    assert not bool(out.z_zero[0])
    return float(out.log_z[0])


def _lse(x):
    m = x.max()
    return float(m + np.log(np.exp(x - m).sum()))


def test_lattice_equals_reverse_marginal_at_tau1_and_record_tau2_gap():
    """T1.11: at tau = beta = 1 with a band that never binds (W 60, S 10, T 4) the lattice log Z is
    logsumexp_a [sum log q + sum P + log p_phi(z | B(a))] (1e-4 abs). At tau = 2 the gap between the
    code's form (1/tau on every (path, segmentation) term) and NOTE l.155's (1/tau on the marginal)
    is RECORDED, not asserted (S1); only its sign is a sanity check (sum_seg p^(1/2) >= (sum p)^(1/2))."""
    model, log_q, prior, z, eta = _identity_inputs()
    q_terms, rev_terms = _note_terms(model, log_q, prior, z, eta)
    assert len(q_terms) > 10
    code1 = _lattice_log_z(model, log_q, prior, z, eta, 1.0)
    note1 = _lse(q_terms + rev_terms)
    assert abs(code1 - note1) <= 1e-4, (code1, note1)
    code2 = _lattice_log_z(model, log_q, prior, z, eta, 2.0)
    note2 = _lse(0.5 * (q_terms + rev_terms))
    gap = code2 - note2
    print(f"\nT1.11 RECORD tau=2: code log Z = {code2:.6f}, NOTE l.155 log Z = {note2:.6f}, "
          f"gap (code - NOTE) = {gap:.6f} nats (S = 10, T = 4, W = 60; tau=1 diff {code1 - note1:.2e})")
    assert np.isfinite(gap) and gap >= -1e-4
