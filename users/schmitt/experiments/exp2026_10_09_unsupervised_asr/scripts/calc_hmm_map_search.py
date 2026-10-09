#!/usr/bin/env python3
"""
Task D: the master objective via a frozen-transition HMM (segmented / cheat-seg case).

The n-gram criterion (`calc_soft_map_search.py`) truncates the audio-text match at order 4. This
replaces the truncation by a full-length contraction: make the audio side first-order Markov,

    A[x,x'] = p_A(x' | x),  pi[x] = p_A(x),  D_c = diag(M[:, c])

and the probability the induced model assigns to a phoneme sequence is a forward pass

    q(c_1^T) = pi' D_{c1} A D_{c2} A ... A D_{cT} 1 .

The objective is `max_M sum_{c_1^T} p(c_1^T) log q(c_1^T)` over the (disjoint) text corpus. Since
`p` is fixed this is `min_M KL(p || q) + const`, i.e. the SAME forward direction as the n-gram
criterion, now at full sequence length rather than order 4.

Two search modes:
  * EM / Baum-Welch with `A` and `pi` FROZEN, only `M` re-estimated from the state posteriors.
    Monotone in the objective -- the principled attack on "search, not objective, is the bottleneck".
  * gradient on `theta` with the same temperature anneal as the n-gram script, for comparability.

Selection across seeds is by the hardened (argmax) objective only, never by accuracy.

  ./calc_hmm_map_search.py --self-test          # the correctness gate, no data needed
  ./calc_hmm_map_search.py --mode em --init uniform --seeds 4
"""
import argparse
import sys
import time

import numpy as np
import torch

from calc_cheat_seg_identifiability import NUM_PHON, load_data, oracle_table
from calc_soft_map_search import unigram_bigram, trigram_dense, fourgram_dense, SoftMapLoss

EPS = 1e-12


def markov_from_bigram(P2):
    """(pi, A) of the first-order chain whose joint bigram is exactly `P2`."""
    pi = P2.sum(1)
    A = P2 / np.maximum(pi[:, None], EPS)
    dead = pi <= 0
    if dead.any():  # unobserved cluster: leave the chain in place rather than producing a zero row
        A[dead] = 0.0
        A[dead, np.where(dead)[0]] = 1.0
    return pi, A


def smooth_chain(pi, A, eps):
    """Interpolate the transitions with the stationary-ish marginal, so nothing is unreachable."""
    if not eps:
        return pi, A
    return (1 - eps) * pi + eps / len(pi), (1 - eps) * A + eps * pi[None, :]


class HmmObjective:
    """
    The full-length forward pass, batched over padded sequences and scaled per step.

    `batches` is a list of (labels [B, Tmax] int64, lens [B]) with the sequences bucketed by length.
    """

    def __init__(self, pi, A, batches, num_tokens):
        self.pi = torch.as_tensor(pi, dtype=torch.float64)
        self.A = torch.as_tensor(A, dtype=torch.float64)
        self.batches = batches
        self.num_tokens = float(num_tokens)

    def _emissions(self, M, labels):
        """[B, T, n]: M[:, c_t] per step, 1 where the step is padding (handled by the caller)."""
        return M[:, labels].permute(1, 2, 0)  # [n, B, T] -> [B, T, n]

    @torch.no_grad()
    def logq(self, M, labels, lens):
        """Per-sequence log q(c_1^T), scaled forward recursion (no logs inside the recursion)."""
        B, T = labels.shape
        E = self._emissions(M, labels)
        alpha = self.pi[None, :] * E[:, 0, :]
        s = alpha.sum(1)
        out = torch.log(s + EPS)
        alpha = alpha / (s[:, None] + EPS)
        for t in range(1, T):
            nxt = (alpha @ self.A) * E[:, t, :]
            s = nxt.sum(1)
            live = (t < lens).to(alpha.dtype)
            out = out + live * torch.log(s + EPS)
            nxt = nxt / (s[:, None] + EPS)
            alpha = live[:, None] * nxt + (1.0 - live)[:, None] * alpha
        return out

    def __call__(self, M):
        """Corpus cross-entropy per token = KL(p || q) + H(p); the quantity to minimize."""
        tot = 0.0
        for labels, lens in self.batches:
            tot = tot + self.logq(M, labels, lens).sum()
        return -tot / self.num_tokens

    def hard(self, assign, m, floor=1e-6):
        """The objective at the hardened map. A tiny floor keeps unused phonemes from giving -inf."""
        M = torch.full((len(assign), m), floor / m, dtype=torch.float64)
        M[torch.arange(len(assign)), torch.as_tensor(assign)] += 1.0 - floor
        with torch.no_grad():
            return float(self(M))

    @torch.no_grad()
    def em_counts(self, M):
        """
        Baum-Welch E-step with `pi`, `A` frozen: returns the expected (state, phoneme) counts.
        `M`-step is then a row normalization of those counts.
        """
        n, m = M.shape
        J = torch.zeros(n, m, dtype=torch.float64)
        ll = 0.0
        for labels, lens in self.batches:
            B, T = labels.shape
            E = self._emissions(M, labels)
            live = (torch.arange(T)[None, :] < lens[:, None]).to(M.dtype)  # [B, T]
            # forward
            ah = torch.empty(B, T, n, dtype=torch.float64)
            sc = torch.empty(B, T, dtype=torch.float64)
            a = self.pi[None, :] * E[:, 0, :]
            sc[:, 0] = a.sum(1)
            ah[:, 0] = a / (sc[:, 0][:, None] + EPS)
            for t in range(1, T):
                a = (ah[:, t - 1] @ self.A) * E[:, t, :]
                sc[:, t] = a.sum(1) * live[:, t] + (1.0 - live[:, t])
                ah[:, t] = torch.where(live[:, t][:, None].bool(),
                                       a / (sc[:, t][:, None] + EPS), ah[:, t - 1])
            ll += float((live * torch.log(sc + EPS)).sum())
            # backward (same scaling factors)
            bh = torch.ones(B, n, dtype=torch.float64)
            for t in range(T - 1, -1, -1):
                g = ah[:, t] * bh
                g = g / (g.sum(1, keepdim=True) + EPS) * live[:, t][:, None]
                J.scatter_add_(1, labels[:, t][None, :].expand(n, B), g.t())
                if t > 0:
                    nb = (E[:, t, :] * bh) @ self.A.t() / (sc[:, t][:, None] + EPS)
                    bh = torch.where(live[:, t][:, None].bool(), nb, bh)
        return J, -ll / self.num_tokens


    @torch.no_grad()
    def grad_wrt_M(self, M):
        """
        (dL/dM, L, J) without autograd. `log q` is linear in each emission along a path, so
        d log q / d M[x,c] = (expected number of times state x emits c) / M[x,c] -- i.e. the same
        Baum-Welch counts the EM step uses. One E-step per gradient step, instead of taping a
        T-step recursion.
        """
        J, nll = self.em_counts(M)
        return -J / (self.num_tokens * (M + EPS)), nll, J


def make_batches(seqs, batch_size):
    """Pad into length-bucketed batches of (labels [B, T], lens [B])."""
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i]))
    out = []
    for k in range(0, len(order), batch_size):
        chunk = [seqs[i] for i in order[k:k + batch_size]]
        T = max(len(c) for c in chunk)
        lab = np.zeros((len(chunk), T), dtype=np.int64)
        for i, c in enumerate(chunk):
            lab[i, : len(c)] = c
        out.append((torch.as_tensor(lab), torch.as_tensor([len(c) for c in chunk])))
    return out


# ---------------------------------------------------------------- the correctness gate


def truncation_test(seed=0, n=7, m=4, tol=1e-10):
    """
    THE correctness test for Task D: under a Markov `p_A`, truncating the forward product to 2, 3
    and 4 factors must reproduce the existing multilinear contractions `q[2]`, `q[3]`, `q[4]` of
    `calc_soft_map_search.SoftMapLoss` exactly. Runs on random tables, no data needed.
    """
    rng = np.random.default_rng(seed)
    pi = rng.random(n); pi /= pi.sum()
    A = rng.random((n, n)); A /= A.sum(1, keepdims=True)
    M = rng.random((n, m)); M /= M.sum(1, keepdims=True)
    Mt = torch.as_tensor(M)

    # the Markov chain's own n-gram tables, in the formats the existing code expects
    P2 = pi[:, None] * A
    P3 = P2[:, :, None] * A[None, :, :]
    P4 = P3[:, :, :, None] * A[None, None, :, :]
    idx = np.arange(n)
    C3 = ((idx[:, None, None] * n + idx[None, :, None]).repeat(n, 2).reshape(-1),
          np.broadcast_to(idx[None, None, :], (n, n, n)).reshape(-1), P3.reshape(-1))
    pc = ((idx[:, None, None] * n + idx[None, :, None]) * n + idx[None, None, :]).reshape(-1)
    C4 = (pc // n, pc % n, np.arange(n ** 3).repeat(n),
          np.broadcast_to(idx, (n ** 3, n)).reshape(-1), P4.reshape(-1))

    t1 = np.full(m, 1.0 / m)
    t2 = np.full((m, m), 1.0 / m ** 2)
    loss = SoftMapLoss(pi, P2, t1, t2, lam=1.0, C3=C3, t3=np.full((m,) * 3, m ** -3.0), tri_scale=1.0,
                       tri_backoff=0.0, C4=C4, t4=np.full((m,) * 4, m ** -4.0), four_scale=1.0,
                       four_backoff=0.0)
    q1, q2 = loss.induced(Mt)
    q3 = loss.induced_trigram(Mt, q1, q2)
    q4 = loss.induced_fourgram(Mt, q1, q2, q3)

    # float64 references straight from the dense Markov tables, so the 4-gram comparison is not
    # limited by `induced_fourgram`'s deliberate float32 stages (~1e-9 on these magnitudes)
    q3_64 = torch.einsum("ijk,ia,jb,kc->abc", torch.as_tensor(P3), Mt, Mt, Mt)
    q4_64 = torch.einsum("ijkl,ia,jb,kc,ld->abcd", torch.as_tensor(P4), Mt, Mt, Mt, Mt)

    obj = HmmObjective(pi, A, [], 1.0)
    ok = True
    for T, ref, name, atol in ((2, q2, "q[2]", tol), (3, q3, "q[3]", tol), (4, q4, "q[4]", 1e-8),
                               (3, q3_64, "float64 einsum", tol), (4, q4_64, "float64 einsum", tol)):
        tup = torch.as_tensor(np.array(np.meshgrid(*([np.arange(m)] * T), indexing="ij"))
                              .reshape(T, -1).T)
        got = torch.exp(obj.logq(Mt, tup, torch.full((tup.shape[0],), T))).reshape((m,) * T)
        err = float((got - ref).abs().max())
        print("  truncate to %d factors vs %-14s: max abs diff %.3e   %s"
              % (T, name, err, "OK" if err < atol else "FAIL"))
        ok &= err < atol
    # the scaled recursion itself, against a naive dense product on random sequences
    lens = rng.integers(3, 30, 8)
    lab = np.zeros((8, lens.max()), dtype=np.int64)
    for i, L in enumerate(lens):
        lab[i, :L] = rng.integers(0, m, L)
    got = obj.logq(Mt, torch.as_tensor(lab), torch.as_tensor(lens))
    for i, L in enumerate(lens):
        v = pi * M[lab[i, 0], :][0] if False else pi * M[:, lab[i, 0]]
        for t in range(1, L):
            v = (v @ A) * M[:, lab[i, t]]
        err = abs(float(got[i]) - float(np.log(v.sum())))
        ok &= err < 1e-9
    print("  scaled recursion vs naive dense product over %d random sequences: max err %.3e"
          % (len(lens), err))
    # normalization: sum over all length-T phoneme sequences must be 1
    for T in (2, 3, 4):
        tup = torch.as_tensor(np.array(np.meshgrid(*([np.arange(m)] * T), indexing="ij"))
                              .reshape(T, -1).T)
        tot = float(torch.exp(obj.logq(Mt, tup, torch.full((tup.shape[0],), T))).sum())
        print("  sum_{c_1^%d} q = %.12f" % (T, tot))
        ok &= abs(tot - 1.0) < 1e-10
    return ok


def run_em(obj, M, steps, acc, m, log_every, label):
    """EM with `pi`, `A` frozen: the M-step is a row normalization of the Baum-Welch counts."""
    t0 = time.time()
    for it in range(steps):
        J, nll = obj.em_counts(M)
        assign = M.argmax(1).numpy()
        if it % log_every == 0 or it == steps - 1:
            print("    %4d  soft %.5f  hard %.5f  acc %.4f  used %2d/%d"
                  % (it, nll, obj.hard(assign, m), acc(assign), len(set(assign.tolist())), m))
        M = J / (J.sum(1, keepdim=True) + EPS)
    return M, time.time() - t0


def run_grad(obj, theta, steps, acc, m, log_every, label, lr, tau_start, tau_end):
    """Adam on the logits with a geometric anneal, using the closed-form gradient."""
    t0 = time.time()
    taus = np.geomspace(tau_start, tau_end, steps)
    opt = torch.optim.Adam([theta], lr=lr)
    for it in range(steps):
        tau = float(taus[it])
        M = torch.softmax(theta.detach() / tau, dim=1)
        gM, nll, _ = obj.grad_wrt_M(M)
        opt.zero_grad()
        theta.grad = (M * (gM - (M * gM).sum(1, keepdim=True))) / tau
        opt.step()
        if it % log_every == 0 or it == steps - 1:
            assign = M.argmax(1).numpy()
            print("    %4d  tau %.3f  soft %.5f  hard %.5f  acc %.4f  used %2d/%d"
                  % (it, tau, nll, obj.hard(assign, m), acc(assign), len(set(assign.tolist())), m))
    return torch.softmax(theta / tau_end, dim=1), time.time() - t0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self-test", action="store_true", help="the truncation correctness gate only")
    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--num-cluster-shards", type=int, default=3)
    p.add_argument("--num-audio-utts", type=int, default=6000, help="matches the n-gram script")
    p.add_argument("--num-text-utts", type=int, default=3000,
                   help="text sequences the full-length objective is summed over")
    p.add_argument("--mode", choices=["em", "grad"], default="em")
    p.add_argument("--init", choices=["uniform", "oracle", "random", "oracle-corrupt", "assign"],
                   default="uniform")
    p.add_argument("--corrupt", type=float, default=0.5,
                   help="--init oracle-corrupt: fraction of rows randomized (basin sweep)")
    p.add_argument("--init-assign", default=None,
                   help="--init assign: npz with `assign`+`keep_p`, e.g. the n-gram solution")
    p.add_argument("--save-npz", default=None)
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--steps", type=int, default=60)
    p.add_argument("--smooth", type=float, default=1e-3,
                   help="interpolate A with the marginal, so no text sequence is unreachable")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.02)
    p.add_argument("--noise", type=float, default=1e-3)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--cache", default="/var/tmp/cheat_seg_identifiability_cache.pkl")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    if args.self_test:
        print("=== Task D correctness gate: HMM forward vs the existing n-gram contractions ===")
        ok = truncation_test()
        print("  -> %s" % ("PASS" if ok else "FAIL"))
        sys.exit(0 if ok else 1)

    t0 = time.time()
    clus, phon = load_data(args.num_clusters, args.num_cluster_shards, args.cache)
    common = sorted(set(clus) & set(phon))
    eq = [t for t in common if len(clus[t]) == len(phon[t])]
    J = oracle_table(clus, phon, eq, args.num_clusters)
    cf = J.sum(1) / J.sum()
    Pph = J / np.maximum(J.sum(1, keepdims=True), 1)
    oracle_acc = float((cf * Pph.max(1)).sum())
    chance = float((Pph * cf[:, None]).sum(0).max())

    audio_tags = eq[: args.num_audio_utts]
    aset = set(audio_tags)
    lm_tags = [t for t in common if t not in aset][: args.num_text_utts]
    c1, C2 = unigram_bigram([clus[t] for t in audio_tags], args.num_clusters)
    t1_full, _ = unigram_bigram([phon[t] for t in lm_tags], NUM_PHON)
    keep_p = np.where(t1_full > 0)[0]
    n, m = args.num_clusters, len(keep_p)
    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(m)
    omap = p_pos[J.argmax(1)]
    omap = np.where(omap < 0, int(t1_full[keep_p].argmax()), omap)
    Pk = Pph[:, keep_p]

    def acc(assign):
        return float((cf * Pk[np.arange(n), assign]).sum())

    seqs = [p_pos[phon[t].astype(np.int64)] for t in lm_tags]
    assert all((s >= 0).all() for s in seqs)
    batches = make_batches(seqs, args.batch_size)
    ntok = sum(len(s) for s in seqs)
    pi, A = markov_from_bigram(C2)
    pi, A = smooth_chain(pi, A, args.smooth)
    obj = HmmObjective(pi, A, batches, ntok)

    print("=== setup (Task D: frozen-transition HMM, cheat-seg) ===")
    print("  audio utts %d -> P_A[2] (%d/%d bigram cells nonzero), disjoint text utts %d (%d tokens)"
          % (len(audio_tags), int((C2 > 0).sum()), n * n, len(lm_tags), ntok))
    print("  map %d x %d = %d parameters | transition smoothing %.1e | %d batches"
          % (n, m, n * m, args.smooth, len(batches)))
    print("  oracle acc %.4f -> PER floor %.1f%% | unigram-argmax chance %.4f"
          % (oracle_acc, 100 * (1 - oracle_acc), chance))
    print("  reference: the n-gram criterion at w=(5,5,10,20) reaches oracle-start 0.7335, "
          "loss-selected cold start 0.7203 (RESULTS.md II.B)")
    print("  oracle map: hardened objective %.5f  acc %.4f" % (obj.hard(omap, m), acc(omap)))

    start_assign = None
    if args.init == "assign":
        d = np.load(args.init_assign)
        assert list(d["keep_p"]) == list(keep_p), "the npz was made with a different phoneme set"
        start_assign = d["assign"]
        print("  --init assign: %s (its own acc %.4f)" % (args.init_assign, acc(start_assign)))

    def onehot_logits(a):
        th = np.full((n, m), -2.0)
        th[np.arange(n), a] = 2.0
        return th

    if args.init == "uniform":
        inits = [("oracle", 0)] + [("uniform", sd) for sd in range(1, args.seeds + 1)]
    elif args.init == "oracle-corrupt":
        inits = [("oracle-corrupt", sd) for sd in range(1, args.seeds + 1)]
    else:
        inits = [(args.init, sd) for sd in range(1, max(args.seeds, 1) + 1)]
    results = []
    for kind, sd in inits:
        rng = np.random.default_rng(sd)
        if kind == "oracle":
            th = onehot_logits(omap)
        elif kind == "assign":
            th = onehot_logits(start_assign)
        elif kind == "oracle-corrupt":
            a = omap.copy()
            sel = rng.random(n) < args.corrupt
            a[sel] = rng.integers(0, m, int(sel.sum()))
            th = onehot_logits(a)
        elif kind == "uniform":
            th = rng.normal(0.0, args.noise, (n, m))
        else:
            th = rng.normal(0.0, 2.0, (n, m))
        theta = torch.tensor(th, dtype=torch.float64, requires_grad=True)
        label = kind if kind in ("oracle", "assign") else "%s seed %d" % (kind, sd)
        if kind == "oracle-corrupt":
            label = "corrupt %.1f seed %d" % (args.corrupt, sd)
        print("\n=== %s (%s) ===" % (label, args.mode))
        if args.mode == "em":
            M0 = torch.softmax(theta.detach() / (1.0 if kind != "oracle" else 0.5), dim=1)
            M, dt = run_em(obj, M0, args.steps, acc, m, args.log_every, label)
        else:
            pass
            M, dt = run_grad(obj, theta, args.steps, acc, m, args.log_every, label,
                             args.lr, args.tau_start, args.tau_end)
        assign = M.argmax(1).numpy()
        hard, a = obj.hard(assign, m), acc(assign)
        print("  [%s] hardened objective %.5f | acc %.4f -> PER %.1f%% | %.0fs"
              % (label, hard, a, 100 * (1 - a), dt))
        results.append(dict(label=label, hard=hard, acc=a, assign=assign,
                            init_acc=acc(th.argmax(1))))

    if args.save_npz:
        b = min(results, key=lambda r: r["hard"])
        # `assign` is the loss-selected map (what another script's --init-assign wants); the stacked
        # arrays keep every run, so a cold-start map can be picked out without re-running.
        np.savez(args.save_npz, assign=b["assign"], keep_p=keep_p, hard=b["hard"], acc=b["acc"],
                 assigns=np.stack([r["assign"] for r in results]),
                 labels=np.array([r["label"] for r in results]),
                 hards=np.array([r["hard"] for r in results]),
                 accs=np.array([r["acc"] for r in results]))
        print("\n  wrote %s (%s, hard %.5f, acc %.4f)" % (args.save_npz, b["label"], b["hard"], b["acc"]))

    print("\n=== summary (sorted by the hardened objective, the only selection criterion) ===")
    print("  %-20s %10s %9s %8s %8s" % ("init", "hard", "init_acc", "acc", "PER %"))
    for r in sorted(results, key=lambda r: r["hard"]):
        print("  %-20s %10.5f %9.4f %8.4f %8.1f"
              % (r["label"], r["hard"], r["init_acc"], r["acc"], 100 * (1 - r["acc"])))
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
