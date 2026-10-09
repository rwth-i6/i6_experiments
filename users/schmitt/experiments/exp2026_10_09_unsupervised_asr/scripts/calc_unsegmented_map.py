#!/usr/bin/env python3
"""
Task C applied to the REAL, unsegmented data: learn a 128 -> phoneme map for the actual clus128
audio tokens (2.6 tokens per phoneme, no oracle segmentation) from unpaired n-gram statistics.

Difference to `calc_soft_map_search.py`, which runs on zyang's oracle GMM segment clustering:
there one audio token is one phoneme, so the map can be scored frame-by-frame against an oracle
table. Here there is no alignment and no oracle map, so

  * the criterion is the SAME forward-KL distribution matching with the collapse-repeats
    pushforward B appended (`SoftMapLoss(collapse=True)`, verified against a direct count), and
  * the result is scored the only way it can be: decode a held-out PAIRED set (map every cluster
    id, collapse repeats) and take the edit distance to the true phoneme sequence -> PER.

Three reference points are computed on the same eval set:
  * random map                       -- the floor,
  * supervised memoryless ceiling    -- the best 128 -> phoneme table obtainable WITH labels, by
                                        Viterbi-EM forced alignment under the same collapse model
                                        (this is the analogue of the 0.740 / 26.0% cheat-seg oracle),
  * the unsupervised, loss-selected map.

Selection across seeds is by hardened criterion only, never by PER.

  ./calc_unsegmented_map.py --seeds 4 --lam-z 10
"""
import argparse
import sys
import time

import numpy as np
import torch

from calc_cheat_seg_identifiability import NUM_PHON, _load_hdfs, PHONEME_HDF
from calc_corpus_diagnostics import CLUS128_HDF, read_vocab, VOCAB
from calc_soft_map_search import SoftMapLoss, unigram_bigram, optimize


def collapse(x):
    """Drop adjacent repeats."""
    if len(x) == 0:
        return x
    return x[np.concatenate([[True], x[1:] != x[:-1]])]


def edit_distance(a, b):
    """
    Levenshtein distance, one vectorized row per symbol of `a`.

    The within-row insertion recurrence cur[j] = min(d[j], cur[j-1] + 1) unrolls to the min-plus
    scan cur[j] = min_{k<=j} (d[k] + (j-k)), i.e. a cumulative minimum of `d - arange` -- so the
    whole row is O(len(b)) numpy work and only the outer loop stays in python.
    """
    b = np.asarray(b)
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    j = np.arange(len(b) + 1)
    prev = j.copy()
    for i, ca in enumerate(a, 1):
        d = np.empty(len(b) + 1, dtype=np.int64)
        d[0] = i
        d[1:] = np.minimum(prev[1:] + 1, prev[:-1] + (b != ca))
        prev = np.minimum.accumulate(d - j) + j
    return int(prev[-1])


def per(assign, tags, clus, phon):
    """(PER, mean hyp len, mean ref len) of `assign` on a paired eval set."""
    err = ref = 0
    lh = lr = 0
    for t in tags:
        hyp = collapse(assign[clus[t].astype(np.int64)])
        r = phon[t].astype(np.int64)
        err += edit_distance(hyp, r)
        ref += len(r)
        lh += len(hyp)
        lr += len(r)
    return 100.0 * err / ref, lh / len(tags), lr / len(tags)


def per_shuffled(assign, tags, clus, phon):
    """
    The shuffle control of `calc_shuffle_control.py`, in PER form: score each hypothesis against a
    DIFFERENT, length-matched utterance's reference (rotation within a length-sorted order). An
    edit alignment between two strings from the same unigram distribution matches tokens by chance,
    so only `shuffled PER - matched PER` is evidence that the map carries acoustic content.
    """
    order = sorted(tags, key=lambda t: len(phon[t]))
    err = ref = 0
    for t, u in zip(order, order[1:] + order[:1]):
        err += edit_distance(collapse(assign[clus[t].astype(np.int64)]), phon[u].astype(np.int64))
        ref += len(phon[u])
    return 100.0 * err / ref


def proportional_counts(tags, clus, phon, n, m):
    """(cluster, phoneme) counts under the position-proportional alignment frame t -> c[t*S/T]."""
    J = np.zeros((n, m))
    for t in tags:
        x = clus[t].astype(np.int64)
        c = phon[t].astype(np.int64)
        J[x, c[(np.arange(len(x)) * len(c)) // len(x)]] += 1
    return J


def viterbi_align_counts(logM, tags, clus, phon, n, m):
    """
    (cluster, phoneme) counts from a forced alignment: each frame emits its phoneme, each phoneme
    takes >= 1 frame (the Viterbi version of `collapse_nll`'s recursion).
    """
    J = np.zeros((n, m))
    for t in tags:
        x = clus[t].astype(np.int64)
        c = phon[t].astype(np.int64)
        T, S = len(x), len(c)
        if T < S:
            continue
        em = logM[np.ix_(x, c)]
        V = np.full(S, -np.inf)
        V[0] = em[0, 0]
        bp = np.zeros((T, S), dtype=bool)
        for i in range(1, T):
            shift = np.concatenate([[-np.inf], V[:-1]])
            take = shift > V
            bp[i] = take
            V = np.where(take, shift, V) + em[i]
        si = S - 1
        for i in range(T - 1, -1, -1):
            J[x[i], c[si]] += 1
            if bp[i, si]:
                si -= 1
    return J


def _pad_batch(tags, clus, phon):
    X = [clus[t].astype(np.int64) for t in tags]
    C = [phon[t].astype(np.int64) for t in tags]
    Tm, Sm = max(len(x) for x in X), max(len(c) for c in C)
    Xp = np.zeros((len(tags), Tm), dtype=np.int64)
    Cp = np.zeros((len(tags), Sm), dtype=np.int64)
    for i, (x, c) in enumerate(zip(X, C)):
        Xp[i, : len(x)] = x
        Cp[i, : len(c)] = c
    return (torch.as_tensor(Xp), torch.as_tensor(Cp),
            torch.as_tensor([len(x) for x in X]), torch.as_tensor([len(c) for c in C]))


def collapse_nll(logM, X, C, Xlen, Clen):
    """
    -log p(c_1^S | x_1^T) summed over the batch under the collapse-repeats model.

    Mapping every frame through `M` and then collapsing adjacent repeats is exactly CTC WITHOUT a
    blank symbol, so the marginal over alignments is the forward recursion

        alpha[t, s] = ( alpha[t-1, s] + alpha[t-1, s-1] ) * M[x_t, c_s]

    (stay in the current phoneme, or advance one). Reference sequences with an adjacent repeat are
    unreachable under this model; they are dropped by the caller.
    """
    B, Tm = X.shape
    Sm = C.shape[1]
    s_idx = torch.arange(Sm)[None, :]
    invalid = s_idx >= Clen[:, None]
    neg = torch.finfo(logM.dtype).min / 4

    def emit(t):
        return logM[X[:, t][:, None], C]  # [B, Sm]

    la = torch.full((B, Sm), neg, dtype=logM.dtype)
    la = torch.cat([emit(0)[:, :1], la[:, 1:]], dim=1)
    for t in range(1, Tm):
        shift = torch.cat([torch.full((B, 1), neg, dtype=logM.dtype), la[:, :-1]], dim=1)
        nxt = torch.logaddexp(la, shift) + emit(t)
        nxt = nxt.masked_fill(invalid, neg)
        la = torch.where((t < Xlen)[:, None], nxt, la)
    return -la.gather(1, (Clen - 1)[:, None]).squeeze(1)


def supervised_map(tags, clus, phon, n, m, epochs=20, batch=192, lr=0.1, log=print):
    """
    The supervised ceiling: maximum-likelihood fit of the SAME 128 x m memoryless map under the
    SAME collapse model (`collapse_nll`), by Adam on its logits. This is the best a table of this
    shape can do with labels, and the analogue of the cheat-seg oracle's 0.740 / 26.0% PER.
    """
    # the model cannot emit an adjacent repeat, so the reference it is fitted to is the collapsed
    # one (the analogue of the criterion's diagonal back-off). ~50% of utterances contain at least
    # one geminate, so dropping them instead would throw away half the corpus.
    ref = {t: collapse(phon[t].astype(np.int64)).astype(np.int16) for t in tags}
    tags = [t for t in tags if len(clus[t]) >= len(ref[t])]
    order = sorted(range(len(tags)), key=lambda i: len(clus[tags[i]]))  # bucket by length
    batches = [_pad_batch([tags[i] for i in order[k:k + batch]], clus, ref)
               for k in range(0, len(order), batch)]
    ntok = float(sum(len(ref[t]) for t in tags))
    logits = torch.zeros(n, m, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        logits += torch.as_tensor(np.random.default_rng(0).normal(0, 1e-2, (n, m)))
    opt = torch.optim.Adam([logits], lr=lr)
    for ep in range(epochs):
        tot = 0.0
        for X, C, Xl, Cl in batches:
            opt.zero_grad()
            nll = collapse_nll(torch.log_softmax(logits, dim=1), X, C, Xl, Cl).sum()
            nll.backward()
            opt.step()
            tot += float(nll.detach())
        log("  epoch %2d: NLL/token %.4f" % (ep, tot / ntok))
    return logits.detach(), len(tags)


def per_coordinate_descent(assign, fit_tags, clus, phon, m, order, sweeps=2, log=print):
    """
    The supervised ceiling, defined operationally: hill-climb the table against the metric that is
    actually reported (PER on a held-in paired set), one cluster at a time. Labels are used
    throughout -- this is a reference for the unsupervised numbers, never a candidate result.
    """
    cur = np.asarray(assign).copy()
    base = per(cur, fit_tags, clus, phon)[0]
    for sw in range(sweeps):
        changed = 0
        for k in order:
            old = int(cur[k])
            best_v, best_e = old, base
            for v in range(m):
                if v == old:
                    continue
                cur[k] = v
                e = per(cur, fit_tags, clus, phon)[0]
                if e < best_e - 1e-9:
                    best_e, best_v = e, v
            cur[k] = best_v
            base = best_e
            changed += best_v != old
        log("  sweep %d: fit-set PER %.2f%% (%d/%d clusters moved)" % (sw, base, changed, len(cur)))
        if changed == 0:
            break
    return cur


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--clus128-shards", type=int, default=4, help="of 10, each ~28k utts")
    p.add_argument("--num-audio-utts", type=int, default=20000)
    p.add_argument("--num-text-utts", type=int, default=60000)
    p.add_argument("--num-eval-utts", type=int, default=1000)
    p.add_argument("--num-ceiling-utts", type=int, default=3000)
    p.add_argument("--em-iters", type=int, default=20, help="epochs of the supervised ML fit")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--per-cd-sweeps", type=int, default=2, help="supervised PER hill-climb; 0 = off")
    p.add_argument("--num-per-cd-utts", type=int, default=150)
    p.add_argument("--lam", type=float, default=5.0)
    p.add_argument("--lam-z", type=float, default=10.0)
    p.add_argument("--z-target", type=float, default=None)
    p.add_argument("--no-collapse", action="store_true", help="ablation: drop B (then Z is unconstrained)")
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.02)
    p.add_argument("--init-scale", type=float, default=2.0)
    p.add_argument("--entropy-scale", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-npz", default=None, help="write the loss-selected map + scores")
    args = p.parse_args()
    args.tri_scale = args.four_scale = 0.0

    t0 = time.time()
    clus = _load_hdfs([CLUS128_HDF % i for i in range(args.clus128_shards)])
    phon = _load_hdfs([PHONEME_HDF % i for i in range(10)])
    common = sorted(set(clus) & set(phon))
    if not common:
        sys.exit("no tags in common")
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(len(common))
    common = [common[i] for i in perm]

    # disjoint splits: the audio statistics, the text statistics, the paired eval set and the
    # (supervised-only) set the ceiling is estimated on never share an utterance.
    a0 = args.num_audio_utts
    t0_ = a0 + args.num_text_utts
    e0 = t0_ + args.num_eval_utts
    audio_tags = common[:a0]
    text_tags = common[a0:t0_]
    eval_tags = common[t0_:e0]
    ceil_tags = common[e0:e0 + args.num_ceiling_utts]
    assert eval_tags and ceil_tags, "not enough utterances -- raise --clus128-shards"

    n = 128
    c1, C2 = unigram_bigram([clus[t] for t in audio_tags], n)
    t1_full, t2_full = unigram_bigram([phon[t] for t in text_tags], NUM_PHON)
    keep_p = np.where(t1_full > 0)[0]
    t1 = t1_full[keep_p] / t1_full[keep_p].sum()
    t2 = t2_full[np.ix_(keep_p, keep_p)]
    t2 = t2 / t2.sum()
    m = len(keep_p)
    inv = read_vocab(VOCAB)

    # phoneme ids -> compact 0..m-1, so the eval reference lives in the same index space as the map
    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(m)
    phon = {t: p_pos[v.astype(np.int64)].astype(np.int16) for t, v in phon.items()}
    assert all((phon[t] >= 0).all() for t in eval_tags), "eval reference has a phoneme absent from the text split"

    mean_s = float(np.mean([len(phon[t]) for t in text_tags]))
    mean_t = float(np.mean([len(clus[t]) for t in audio_tags]))
    n_tok = sum(len(clus[t]) for t in audio_tags)
    bigram_ratio = sum(max(len(clus[t]) - 1, 0) for t in audio_tags) / n_tok
    z_target = (mean_s / mean_t) if args.z_target is None else args.z_target

    print("=== setup (real clus128, unsegmented) ===")
    print("  utts: %d audio / %d text / %d eval / %d ceiling (disjoint), %d loaded"
          % (len(audio_tags), len(text_tags), len(eval_tags), len(ceil_tags), len(common)))
    print("  map: %d clusters x %d phonemes = %d parameters" % (n, m, n * m))
    print("  label-free length target: E[S] %.2f / E[T] %.2f -> R = %.3f, z_target %.4f (lam_z %.3g)"
          % (mean_s, mean_t, mean_t / mean_s, z_target, args.lam_z))

    loss_fn = SoftMapLoss(c1, C2, t1, t2, lam=args.lam, entropy_scale=args.entropy_scale,
                          collapse=not args.no_collapse, lam_z=args.lam_z, z_target=z_target,
                          bigram_ratio=bigram_ratio)
    if not args.no_collapse:
        print("  text bigram diagonal backed off: %.5f of the mass dropped + renormalized" % loss_fn.diag_mass)

    # ---------------- supervised ceiling (paired, labels used -- a reference, not a result)
    print("\n=== supervised references on %d paired utts ===" % len(ceil_tags))
    cands = {}
    J = proportional_counts(ceil_tags, clus, phon, n, m)
    cands["proportional alignment"] = J.argmax(1)
    logM = np.log(J / np.maximum(J.sum(1, keepdims=True), 1) + 1e-8)
    cands["+1 Viterbi realignment"] = viterbi_align_counts(logM, ceil_tags, clus, phon, n, m).argmax(1)
    print("  maximum-likelihood fit of the same map under the collapse model:")
    sup_logits, n_sup = supervised_map(ceil_tags, clus, phon, n, m, epochs=args.em_iters,
                                       log=lambda s_: print("  " + s_))
    cands["ML fit (%d utts)" % n_sup] = sup_logits.argmax(dim=1).numpy()
    logM = torch.log_softmax(sup_logits, dim=1).numpy()
    cands["ML + Viterbi realignment"] = viterbi_align_counts(logM, ceil_tags, clus, phon, n, m).argmax(1)
    sup_per, best_a = None, None
    for name, a in cands.items():
        e, lh, _ = per(a, eval_tags, clus, phon)
        print("  %-26s PER %6.2f%%  (hyp %.1f tokens)" % (name, e, lh))
        if sup_per is None or e < sup_per:
            sup_per, best_a = e, a
    if args.per_cd_sweeps:
        fit_tags = ceil_tags[: args.num_per_cd_utts]
        freq = np.bincount(np.concatenate([clus[t].astype(np.int64) for t in fit_tags]), minlength=n)
        print("  hill-climbing PER directly (supervised, %d fit utts, %d sweeps):"
              % (len(fit_tags), args.per_cd_sweeps))
        cd = per_coordinate_descent(best_a, fit_tags, clus, phon, m, np.argsort(-freq),
                                    sweeps=args.per_cd_sweeps, log=print)
        e, lh, _ = per(cd, eval_tags, clus, phon)
        print("  %-26s PER %6.2f%%  (hyp %.1f tokens)" % ("PER hill-climb", e, lh))
        sup_per = min(sup_per, e)

    # ---------------- reference points
    rnd = rng.integers(0, m, n)
    rnd_per, rnd_lh, _ = per(rnd, eval_tags, clus, phon)
    uni = int(np.bincount(np.concatenate([phon[t] for t in text_tags]), minlength=m).argmax())
    print("\n=== reference points on the eval set ===")
    print("  random map        PER %.2f%%  (hyp %.1f tokens)" % (rnd_per, rnd_lh))
    print("  supervised map    PER %.2f%%  <- the ceiling this model class can reach" % sup_per)
    print("  (most frequent text phoneme: %s)" % inv[int(keep_p[uni])])

    # ---------------- the unsupervised search
    print("\n=== unsupervised search (uniform init, %d seeds, %d steps) ===" % (args.seeds, args.steps))

    def dummy_acc(assign):  # no oracle table exists here; PER is computed after, never used for selection
        return float("nan")

    results = []
    for sd in range(1, args.seeds + 1):
        g = np.random.default_rng(sd)
        torch.manual_seed(sd)
        logits = torch.as_tensor(g.normal(0.0, 1e-3, (n, m)), dtype=torch.float64)
        print("\n--- seed %d ---" % sd)
        assign, rep = optimize(loss_fn, logits, args, dummy_acc, "seed-%d" % sd)
        e, lh, lr = per(assign, eval_tags, clus, phon)
        z = loss_fn.z_val
        rep.update(seed=sd, per=e, hyp_len=lh, ref_len=lr, z=z)
        print("  seed %d: hardened loss %.4f | Z %.4f | PER %.2f%% (hyp %.1f / ref %.1f)"
              % (sd, rep["hard"], z, e, lh, lr))
        results.append(rep)

    best = min(results, key=lambda r: r["hard"])
    print("\n=== summary ===")
    print("  %-8s %10s %8s %8s %9s" % ("seed", "hard loss", "Z", "PER %", "hyp len"))
    for r in sorted(results, key=lambda r: r["hard"]):
        print("  %-8d %10.4f %8.4f %8.2f %9.1f%s"
              % (r["seed"], r["hard"], r["z"], r["per"], r["hyp_len"],
                 "   <- loss-selected" if r is best else ""))
    shuf = per_shuffled(best["assign"], eval_tags, clus, phon)
    print("\n  loss-selected: PER %.2f%%   (random map %.2f%%, supervised %.2f%%, ref %.1f tokens/utt)"
          % (best["per"], rnd_per, sup_per, results[0]["ref_len"]))
    print("  shuffle control: same map scored against length-matched WRONG references -> PER %.2f%%"
          "  (gap %+.2f points; ~0 means the output is independent of the audio)"
          % (shuf, shuf - best["per"]))
    if args.save_npz:
        np.savez(args.save_npz, assign=best["assign"], seed=best["seed"], hard=best["hard"],
                 per=best["per"], per_shuffled=shuf, keep_p=keep_p)
        print("  wrote %s" % args.save_npz)
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
