#!/usr/bin/env python3
"""
Task A of the plan (see OBJECTIVES.md / OBJECTIVE_VARIANTS.md): label-free corpus diagnostics that
gate the unsegmented port (Task C) and the frozen-transition HMM (Task D). Nothing here needs a map `M`.

  A1  geminate mass      delta = sum_c p2[c,c] on the text side. Under collapse-repeats every table has
                         q2B[c,c] = 0, so delta > 0 makes the forward KL infinite for all M (Task C blocker).
  A2  compression ratio  R = E[T] / E[S], mean audio-token length over mean phoneme length, from the two
                         UNPAIRED corpora (the label-free target for Z = 1 - sum_c q2y[c,c] in Task C).
                         Reported for the cheat-seg clusters (sanity, ~1) and the real clus128 tokens.
  A3  audio Markov loss  what a first-order p_A discards: KL(P_A[3] || pi p(x2|x1) p(x3|x2)) etc. With
                         consistent marginals this equals the conditional-entropy drop
                         H(x3|x2) - H(x3|x1,x2) = I(x1;x3|x2), so A3 and A4 are the same quantity on the two
                         sides and directly comparable.
  A4  text cond. entropy H(c_k | c_1^{k-1}) for k = 1..5 and the drops between orders.

In-sample (maximum-likelihood) conditional entropies are biased DOWN at high order (an unseen-context
model memorizes), which inflates the drops -- badly so for audio (452k distinct 4-grams over 558k
tokens). So every drop is also reported 2-fold HELD-OUT: fit an interpolated back-off n-gram on one half
of the utterances, score the other half, average both directions. Smoothing = Witten-Bell (parameter
free) plus fixed-weight Jelinek-Mercer interpolation at several weights, to show the sensitivity.

Same data split as calc_soft_map_search.py (cheat-seg: first 6000 equal-length utts = audio side,
disjoint utts = text side), so the numbers describe exactly the statistics the searches use.
"""
import argparse
import sys
import time

import numpy as np

from calc_cheat_seg_identifiability import NUM_PHON, load_data, _load_hdfs

CLUS128_HDF = (
    "/work/asr4/schmitt/sisyphus_work_dirs/2025_10_02_unsup_asr_shared_enc/i6_experiments/users/schmitt/"
    "experiments/exp2025_10_02_shared_enc/librispeech/data/wav2vec/DumpClusterIndicesToHdfJob.yDlHfJxn6vfO/"
    "output/data_%d.hdf"
)
VOCAB = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_experiments/users/schmitt/datasets/utils/"
    "phonemize/PhonemizeTextDataJob.rnyWlvnkNbNd/output/phoneme_vocab.txt"
)


def read_vocab(path):
    import ast
    return {v: k for k, v in ast.literal_eval(open(path).read()).items()}


# ---------------------------------------------------------------- n-gram machinery


def ngram_codes(seqs, V, k):
    """int64 codes of all k-grams (within utterances) as (history_code, next_symbol)."""
    hs, cs = [], []
    for s in seqs:
        s = s.astype(np.int64)
        if len(s) < k:
            continue
        h = np.zeros(len(s) - k + 1, dtype=np.int64)
        for j in range(k - 1):
            h = h * V + s[j : len(s) - k + 1 + j]
        hs.append(h)
        cs.append(s[k - 1 :])
    return np.concatenate(hs), np.concatenate(cs)


def cond_entropy_insample(seqs, V, k):
    """H(x_k | x_1^{k-1}) in nats, maximum likelihood over the k-grams of `seqs`."""
    h, c = ngram_codes(seqs, V, k)
    hc = h * V + c
    _, n_hc = np.unique(hc, return_counts=True)
    _, n_h = np.unique(h, return_counts=True)
    N = len(hc)
    return float(-(n_hc * np.log(n_hc)).sum() / N + (n_h * np.log(n_h)).sum() / N)


class BackoffLM:
    """Interpolated back-off n-gram LM: p_k(c|h) = lam(h) ML_k(c|h) + (1-lam(h)) p_{k-1}(c|h[1:]).
    `mode` = 'wb' (Witten-Bell lam = N(h)/(N(h)+T(h))) or a float (fixed Jelinek-Mercer weight)."""

    def __init__(self, seqs, V, order, mode="wb"):
        self.V, self.order, self.mode = V, order, mode
        self.tables = {}
        for k in range(1, order + 1):
            h, c = ngram_codes(seqs, V, k)
            hc = h * V + c
            u_hc, n_hc = np.unique(hc, return_counts=True)
            u_h, n_h = np.unique(h, return_counts=True)
            # T(h) = number of distinct continuations of h
            t_h = np.bincount(np.searchsorted(u_h, u_hc // V), minlength=len(u_h))
            self.tables[k] = (u_hc, n_hc.astype(float), u_h, n_h.astype(float), t_h.astype(float))

    def _lookup(self, u, n, codes):
        idx = np.searchsorted(u, codes)
        idx = np.minimum(idx, len(u) - 1)
        hit = u[idx] == codes
        return np.where(hit, n[idx], 0.0)

    def logprob(self, seqs, k=None):
        """mean -log p_k(c|h) over the k-gram positions of `seqs` (k defaults to the model order)."""
        k = k or self.order
        h, c = ngram_codes(seqs, self.V, k)
        p = np.full(len(c), 1.0 / self.V)  # order-0 floor: uniform
        for j in range(1, k + 1):
            # the order-j history is the last j-1 symbols of h
            hj = h % (self.V ** (j - 1)) if j > 1 else np.zeros_like(h)
            u_hc, n_hc, u_h, n_h, t_h = self.tables[j]
            n_pair = self._lookup(u_hc, n_hc, hj * self.V + c)
            idx = np.minimum(np.searchsorted(u_h, hj), len(u_h) - 1)
            hit = u_h[idx] == hj
            nh = np.where(hit, n_h[idx], 0.0)
            th = np.where(hit, t_h[idx], 0.0)
            if self.mode == "wb":
                lam = np.where(nh > 0, nh / (nh + th + 1e-12), 0.0)
            else:
                lam = np.where(nh > 0, float(self.mode), 0.0)
            ml = np.where(nh > 0, n_pair / np.maximum(nh, 1e-12), 0.0)
            p = lam * ml + (1.0 - lam) * p
        return float(-np.log(p).mean())


def heldout_curve(seqs, V, max_order, mode, rng):
    """2-fold cross-validated CE per order (positions = k-gram positions of the eval half)."""
    seqs = list(seqs)
    perm = rng.permutation(len(seqs))
    halves = [[seqs[i] for i in perm[: len(seqs) // 2]], [seqs[i] for i in perm[len(seqs) // 2 :]]]
    ce = np.zeros(max_order)
    for a, b in ((0, 1), (1, 0)):
        lm = BackoffLM(halves[a], V, max_order, mode)
        for k in range(1, max_order + 1):
            ce[k - 1] += lm.logprob(halves[b], k) / 2
    return ce


def report_orders(name, seqs, V, max_order, modes, rng):
    print("\n--- %s: conditional entropies H(x_k | x_1^{k-1}) in nats/token, k = 1..%d ---" % (name, max_order))
    ins = np.array([cond_entropy_insample(seqs, V, k) for k in range(1, max_order + 1)])
    rows = [("in-sample ML", ins)]
    for mode in modes:
        rows.append(("held-out %s" % ("Witten-Bell" if mode == "wb" else "JM lam=%.1f" % mode),
                     heldout_curve(seqs, V, max_order, mode, np.random.default_rng(rng.integers(1 << 30)))))
    hdr = "  %-24s" % "" + "".join("%9s" % ("H(k=%d)" % k) for k in range(1, max_order + 1))
    print(hdr)
    for lbl, ce in rows:
        print("  %-24s" % lbl + "".join("%9.4f" % v for v in ce))
    print("  %-24s" % "" + "".join("%9s" % ("drop%d->%d" % (k, k + 1)) for k in range(1, max_order)))
    for lbl, ce in rows:
        print("  %-24s" % lbl + "".join("%9.4f" % (ce[k - 1] - ce[k]) for k in range(1, max_order)))
    return {lbl: ce for lbl, ce in rows}


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--num-cluster-shards", type=int, default=3)
    p.add_argument("--num-audio-utts", type=int, default=6000)
    p.add_argument("--num-lm-utts", type=int, default=60000)
    p.add_argument("--max-order", type=int, default=5)
    p.add_argument("--jm", default="0.3,0.6,0.9", help="fixed JM interpolation weights to sweep")
    p.add_argument("--clus128-shards", type=int, default=2, help="of 10 (each ~28k utts); 0 = skip")
    p.add_argument("--cache", default="/var/tmp/cheat_seg_identifiability_cache.pkl")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)
    modes = ["wb"] + [float(x) for x in args.jm.split(",") if x]
    t0 = time.time()

    clus, phon = load_data(args.num_clusters, args.num_cluster_shards, args.cache)
    common = sorted(set(clus) & set(phon))
    eq = [t for t in common if len(clus[t]) == len(phon[t])]
    audio_tags = eq[: args.num_audio_utts]
    aset = set(audio_tags)
    lm_tags = [t for t in common if t not in aset][: args.num_lm_utts]
    inv = read_vocab(VOCAB)
    print("=== data (same split as calc_soft_map_search.py) ===")
    print("  cheat-seg audio utts %d (%d tokens), disjoint text utts %d (%d tokens)"
          % (len(audio_tags), sum(len(clus[t]) for t in audio_tags), len(lm_tags), sum(len(phon[t]) for t in lm_tags)))

    # ---------------- A1
    text = [phon[t] for t in lm_tags]
    b = np.zeros((NUM_PHON, NUM_PHON))
    for s in text:
        s = s.astype(np.int64)
        np.add.at(b, (s[:-1], s[1:]), 1.0)
    p2 = b / b.sum()
    delta = float(np.trace(p2))
    print("\n=== A1. geminate mass delta = sum_c p2[c,c] (text) ===")
    print("  delta = %.5f  (%.2f%% of bigram tokens; %d of %d bigram tokens)" % (delta, 100 * delta, int(np.trace(b)), int(b.sum())))
    diag = np.diag(p2)
    for c in np.argsort(-diag)[:10]:
        if diag[c] <= 0:
            break
        print("    (%s,%s)  mass %.5f  count %d" % (inv.get(c, c), inv.get(c, c), diag[c], int(b[c, c])))
    # full-corpus check: all text utts of the 10 phoneme shards
    ball = np.zeros((NUM_PHON, NUM_PHON))
    for t in phon:
        s = phon[t].astype(np.int64)
        np.add.at(ball, (s[:-1], s[1:]), 1.0)
    print("  (all %d train-960 text utts: delta = %.5f)" % (len(phon), np.trace(ball) / ball.sum()))
    print("  --> delta > 0: under collapse-repeats q2B[c,c] = 0 for every M, so Task C needs diagonal back-off on p[2]."
          if delta > 0 else "  --> delta = 0: no blocker for Task C.")

    # ---------------- A2
    print("\n=== A2. compression ratio R = E[T] / E[S] (unpaired means) ===")
    la = np.array([len(clus[t]) for t in audio_tags], float)
    ls = np.array([len(phon[t]) for t in lm_tags], float)
    print("  cheat-seg clusters: E[T] %.2f (audio utts), E[S] %.2f (text utts) -> R = %.4f" % (la.mean(), ls.mean(), la.mean() / ls.mean()))
    lp = np.array([len(clus[t]) / len(phon[t]) for t in common])
    print("    paired per-utt T/S over all %d common utts: mean %.4f  std %.4f  p5 %.3f  p50 %.3f  p95 %.3f"
          % (len(common), lp.mean(), lp.std(), *np.percentile(lp, [5, 50, 95])))
    clus128 = None
    if args.clus128_shards:
        clus128 = _load_hdfs([CLUS128_HDF % i for i in range(args.clus128_shards)])
        c128_common = sorted(set(clus128) & set(phon))
        print("  real clus128 tokens (repeats collapsed): %d utts loaded, %d with phoneme text" % (len(clus128), len(c128_common)))
        if not c128_common:
            print("    tag formats: clus128 %r vs phon %r" % (next(iter(clus128)), next(iter(phon))))
        c128_tags = c128_common
        half = len(c128_tags) // 2
        audio128, text128 = c128_tags[:half], c128_tags[half:]  # disjoint utterance sets
        lT = np.array([len(clus128[t]) for t in audio128], float)
        lS = np.array([len(phon[t]) for t in text128], float)
        print("    E[T] %.2f (%d audio utts), E[S] %.2f (%d disjoint text utts) -> R = %.4f"
              % (lT.mean(), len(lT), lS.mean(), len(lS), lT.mean() / lS.mean()))
        r = np.array([len(clus128[t]) / len(phon[t]) for t in c128_common])
        print("    paired per-utt T/S: mean %.4f  std %.4f  p5 %.3f  p50 %.3f  p95 %.3f" % (r.mean(), r.std(), *np.percentile(r, [5, 50, 95])))
        # how much of that is within-phoneme repetition vs. insertion cannot be told without alignment;
        # the collapse model in Task C attributes ALL of it to runs: Z = E[S]/E[T].
        print("    --> collapse-model target Z = E[S]/E[T] = %.4f" % (lS.mean() / lT.mean()))

    # ---------------- A3 / A4
    print("\n=== A3 / A4. conditional entropies and their drops (nats/token) ===")
    print("  KL3 := H(x3|x2) - H(x3|x1,x2)  = drop 2->3 ;  KL4 := KL3 + drop 3->4  (what a Markov-1 audio model discards)")
    audio = [clus[t] for t in audio_tags]
    res = {}
    res["audio"] = report_orders("cheat-seg AUDIO clusters (%d utts, V=%d)" % (len(audio), args.num_clusters), audio, args.num_clusters, args.max_order, modes, rng)
    all_audio = [clus[t] for t in common]
    res["audio_all"] = report_orders("cheat-seg AUDIO clusters, all %d common utts (sampling check)" % len(all_audio), all_audio, args.num_clusters, args.max_order, modes, rng)
    res["text"] = report_orders("TEXT phonemes (%d disjoint utts, V=%d)" % (len(text), NUM_PHON), text, NUM_PHON, args.max_order, modes, rng)
    if clus128 is not None:
        a128 = [clus128[t] for t in audio128]
        res["clus128"] = report_orders("real clus128 AUDIO tokens (%d utts, V=128)" % len(a128), a128, 128, args.max_order, modes, rng)

    print("\n=== decision gate (held-out Witten-Bell; in-sample in brackets) ===")
    def drops(r):
        wb, ins = r["held-out Witten-Bell"], r["in-sample ML"]
        return [(wb[k - 1] - wb[k], ins[k - 1] - ins[k]) for k in range(1, args.max_order)]
    da, dt = drops(res["audio"]), drops(res["text"])
    kl3 = da[1][0]
    kl4 = da[1][0] + da[2][0]
    print("  audio Markov-1 loss: KL3 = %.4f [%.4f]   KL4 = %.4f [%.4f]  nats/token" % (kl3, da[1][1], kl4, da[1][1] + da[2][1]))
    print("  text info gained:    drop 2->3 = %.4f [%.4f]   drop 3->4 = %.4f [%.4f]   drop 4->5 = %.4f [%.4f]"
          % (dt[1][0], dt[1][1], dt[2][0], dt[2][1], dt[3][0], dt[3][1]))
    print("  text H(c) = %.4f, H(c|c') = %.4f ; total text info beyond order 2 (held-out, to order %d) = %.4f"
          % (res["text"]["held-out Witten-Bell"][0], res["text"]["held-out Witten-Bell"][1], args.max_order,
             res["text"]["held-out Witten-Bell"][1] - res["text"]["held-out Witten-Bell"][-1]))
    print("  ratio  audio KL3 / text drop 2->3 = %.2f ;  audio KL4 / text (drop 2->3 + 3->4) = %.2f" % (kl3 / max(dt[1][0], 1e-9), kl4 / max(dt[1][0] + dt[2][0], 1e-9)))
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
