#!/usr/bin/env python3
"""
Is the cheating-GMM-segmentation task (audio cluster id -> phoneme, 1:1) identifiable
from *unpaired* text statistics alone?

Background
----------
`config_librispeech_960_wo_sil_cheat_seg_v1` replaces the noisy k-means segmentation with
zyang's oracle GMM segment clustering, so one audio token = one phoneme and the alignment
problem is gone. What remains is a 512 -> 41 substitution cipher. Every unsupervised model
trained on it still decodes at ~83% PER, and `calc_shuffle_control.py` says all of them are
at chance (matched Corr == shuffled Corr). This script asks why, using only the data.

What it measures
----------------
1. ORACLE CEILING. On the utterances where the cluster and phoneme sequences happen to have
   equal length (25% of them; the clusters are 1.7% shorter on average), positional alignment
   is verified by a shift sweep and the best memoryless 512->41 map is read off. That map's
   frame accuracy is the ceiling for any context-free mapping, and 1 - it is the PER floor.

2. IDENTIFIABILITY of that map from unpaired statistics. A phoneme bigram + unigram
   distribution is estimated from *disjoint* text utterances, and the 512->41 table is then
   searched (hard assignments, coordinate hill-climb, random restarts) to match it. Two
   objective directions are compared, and this is the whole point:

     "ce"  minimize  -sum_ij Q_ij log P_ij      (LM cross-entropy of the induced sequence)
     "kl"  minimize  KL(P || Q)                  (match the text bigram distribution)

   where P = text bigram, Q = bigram induced by mapping the cluster sequences.

Findings (train-other-960, k512 clusters, 6000 audio utts / 60000 LM utts)
--------------------------------------------------------------------------
   oracle memoryless map                       acc 0.740   (PER floor 26.0%)
   unigram-argmax chance                       acc 0.103

   "ce"  objective: oracle loss 3.168 / acc 0.740
                    best random restart 2.980 / acc 0.133
                    hill-climb STARTED AT THE ORACLE: 2.892 / acc 0.480 (229/512 moved)
     -> the LM-cross-entropy objective *strictly prefers wrong maps over the truth*. A lossy
        many-to-one map can make the output MORE LM-likely than real text (real text scores
        2.767). Never use an LM score alone as the unsupervised signal.

   "kl"  objective: oracle loss 0.213 / acc 0.740
                    best random restart 0.206 / acc 0.045
                    hill-climb from the oracle: 0.073 / acc 0.657
                    corrupt-the-oracle basin sweep: loss is monotone in accuracy
                      (0.073@0.66, 0.116@0.48, 0.145@0.38, 0.202@0.20, 0.216@0.12)
     -> distribution matching DOES rank the truth correctly, and by a wide margin (the true
        basin is 3x deeper than any random-restart optimum). It is not reachable from a random
        init: the hill-climb never *increases* accuracy, it only preserves whatever it starts
        with. So for this task the loss is (nearly) right and the SEARCH / INITIALIZATION is
        the bottleneck -- not model capacity, and not the amount of matching signal.

   `models/train_steps/output_stats.py` already uses the "kl" direction (`_kl(acc.bigram,
   pred_bigram)`), so the loss there is the right one; what it lacks in the cheat-seg setup is
   an initialization inside the correct basin.

Usage
-----
    ./calc_cheat_seg_identifiability.py                  # everything, both objectives
    ./calc_cheat_seg_identifiability.py --objective kl   # one objective
    ./calc_cheat_seg_identifiability.py --basin          # add the corrupt-the-oracle sweep
    ./calc_cheat_seg_identifiability.py --num-clusters 1024

Plain script, no sisyphus graph load. Reads the HDFs directly (~1 min) and caches them.
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

CLUSTER_HDF = (
    "/u/zyang/setups/mini/output/example_setups/librispeech/phmm_standalone_2024/"
    "ls960_gmm_oracle_segment_clustering/clustering/k%d/compat/cluster_labels_k%d.%03d.returnn_compat.hdf"
)
# train-other-960 phoneme indices, sil_prob=0.0 (the HDFs the wo_sil configs use)
PHONEME_HDF = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_experiments/users/schmitt/"
    "datasets/utils/phonemize/DumpPhonemeIndicesToHdfJob.V3rHWoSUS9Hf/output/data_%d.hdf"
)
NUM_PHON = 42  # 41 phonemes + the unused slot; harmless, the counts just stay 0


def _load_hdfs(files):
    import h5py

    out = {}
    for fn in files:
        with h5py.File(fn, "r") as f:
            tags = [t.decode() if isinstance(t, bytes) else t for t in f["seqTags"][:]]
            lens = f["seqLengths"][:, 0]
            inputs = f["inputs"][:]
        off = np.concatenate([[0], np.cumsum(lens)])
        for i, t in enumerate(tags):
            out[t] = inputs[off[i] : off[i + 1]].astype(np.int16)
    return out


def load_data(num_clusters, num_cluster_shards, cache):
    if cache and os.path.exists(cache):
        with open(cache, "rb") as f:
            d = pickle.load(f)
        return d["clus"], d["phon"]
    clus = _load_hdfs([CLUSTER_HDF % (num_clusters, num_clusters, i) for i in range(num_cluster_shards)])
    phon = _load_hdfs([PHONEME_HDF % i for i in range(10)])
    if cache:
        with open(cache, "wb") as f:
            pickle.dump({"clus": clus, "phon": phon}, f, protocol=4)
    return clus, phon


def oracle_table(clus, phon, eq_tags, num_clusters):
    """Joint count table [num_clusters, NUM_PHON] over positionally aligned frames."""
    J = np.zeros((num_clusters, NUM_PHON))
    for t in eq_tags:
        np.add.at(J, (clus[t].astype(np.int64), phon[t].astype(np.int64)), 1)
    return J


def report_alignment(clus, phon, common, eq_tags, J, num_clusters):
    lc = np.array([len(clus[t]) for t in common], dtype=float)
    lp = np.array([len(phon[t]) for t in common], dtype=float)
    print("=== 1. is it really 1:1? ===")
    print("  utterances with both modalities: %d" % len(common))
    print("  mean len: clusters %.2f  phonemes %.2f  ratio %.4f" % (lc.mean(), lp.mean(), lc.mean() / lp.mean()))
    print("  exactly equal length: %d (%.1f%%)" % (len(eq_tags), 100.0 * len(eq_tags) / len(common)))
    print("  --> the oracle segmentation is 1:1 only on average; per utterance it is off by a few.")
    print("\n=== 2. is the equal-length subset positionally aligned? (shift sweep) ===")
    for s in (-2, -1, 0, 1, 2):
        K = np.zeros((num_clusters, NUM_PHON))
        for t in eq_tags:
            c, p = clus[t].astype(np.int64), phon[t].astype(np.int64)
            cc, pp = (c[s:], p[: len(p) - s]) if s >= 0 else (c[: len(c) + s], p[-s:])
            np.add.at(K, (cc, pp), 1)
        print("  shift %+d: memoryless-map acc %.4f" % (s, K.max(1).sum() / K.sum()))
    cf = J.sum(1) / J.sum()
    Pph = J / np.maximum(J.sum(1, keepdims=True), 1)
    oracle_acc = float((cf * Pph.max(1)).sum())
    uni = float((Pph * cf[:, None]).sum(0).max())
    print("\n=== 3. oracle ceiling of a memoryless 512->41 map ===")
    print("  oracle acc %.4f  -> PER floor %.1f%%" % (oracle_acc, 100 * (1 - oracle_acc)))
    print("  unigram-argmax chance acc %.4f" % uni)
    used = int((J.sum(1) > 0).sum())
    pur = J.max(1)[J.sum(1) > 0] / J.sum(1)[J.sum(1) > 0]
    print("  clusters used %d/%d, cluster purity mean %.3f median %.3f" % (used, num_clusters, pur.mean(), np.median(pur)))
    return cf, Pph, oracle_acc


class Search:
    """Hard-assignment coordinate hill-climb over the cluster->phoneme table."""

    def __init__(self, Cb, cu, TB, TU, objective, lam=5.0, eps=1e-9, collapse=False, lam_z=0.0,
                 z_target=1.0, bigram_ratio=1.0):
        self.Cb, self.cu, self.TB, self.TU = Cb, cu, TB, TU
        self.objective, self.lam, self.eps = objective, lam, eps
        # Task C: append the parameter-free collapse-repeats map B (see `_score`). q2B has a zero
        # diagonal by construction, so the text bigram's geminate mass is backed off here.
        self.collapse, self.lam_z, self.z_target = bool(collapse), float(lam_z), float(z_target)
        self.bigram_ratio = float(bigram_ratio)
        self.diag_mass = 0.0
        if self.collapse:
            self.diag_mass = float(np.diag(TB).sum())
            TB = TB - np.diag(np.diag(TB))
            TB = TB / TB.sum()
            self.TB = TB
        self.logTB = np.log(TB + eps)
        self.logTU = np.log(TU + eps)
        self.constB = float((TB * self.logTB).sum())
        self.constU = float((TU * self.logTU).sum())
        self.n = Cb.shape[0]

    def _Q(self, m):
        Q = np.zeros((NUM_PHON, NUM_PHON))
        for a in range(self.n):
            np.add.at(Q[m[a]], m, self.Cb[a])
        return Q

    def _pu(self, m):
        pu = np.zeros(NUM_PHON)
        np.add.at(pu, m, self.cu)
        return pu

    def _bi(self, Q):
        if self.objective == "ce":  # -sum_ij Q_ij log P_ij, LM cross-entropy of the output
            return -float((Q * self.logTB).sum())
        return -float((self.TB * np.log(Q + self.eps)).sum()) + self.constB  # KL(P||Q)

    def _un(self, pu):
        return -float((self.TU * np.log(pu + self.eps)).sum()) + self.constU  # always KL

    def _score(self, Q, pu):
        """The criterion on a candidate (induced bigram, induced unigram) pair, collapse included."""
        if not self.collapse:
            return self._bi(Q) + self.lam * self._un(pu)
        d = np.diag(Q).copy()
        r = self.bigram_ratio
        Z = 1.0 - r * d.sum()
        q1B = np.maximum(pu - r * d, 0.0)
        q1B = q1B / (q1B.sum() + self.eps)
        q2B = (Q - np.diag(d)) / (1.0 - d.sum() + self.eps)
        v = self._bi(q2B) + self.lam * self._un(q1B)
        if self.lam_z:
            v += self.lam_z * (Z - self.z_target) ** 2
        return v

    def z_of(self, m):
        """Z = collapsed length / frame length of the sequence the hard map `m` induces."""
        return float(1.0 - self.bigram_ratio * np.diag(self._Q(m)).sum())

    def total(self, m):
        return self._score(self._Q(m), self._pu(m))

    def climb(self, m, rng, sweeps=25):
        m = m.copy()
        for _ in range(sweeps):
            Q, pu = self._Q(m), self._pu(m)
            changed = 0
            for k in rng.permutation(self.n):
                old = int(m[k])
                rvec = np.zeros(NUM_PHON)
                cvec = np.zeros(NUM_PHON)
                np.add.at(rvec, m, self.Cb[k])
                np.add.at(cvec, m, self.Cb[:, k])
                self_ = self.Cb[k, k]
                rvec[old] -= self_
                cvec[old] -= self_
                Qb = Q.copy()
                Qb[old] -= rvec
                Qb[:, old] -= cvec
                Qb[old, old] -= self_
                np.maximum(Qb, 0, out=Qb)
                pub = pu.copy()
                pub[old] -= self.cu[k]
                best_v, best_t = old, None
                for v in range(NUM_PHON):
                    Qv = Qb.copy()
                    Qv[v] += rvec
                    Qv[:, v] += cvec
                    Qv[v, v] += self_
                    p2 = pub.copy()
                    p2[v] += self.cu[k]
                    t = self._score(Qv, p2)
                    if best_t is None or t < best_t - 1e-12:
                        best_t, best_v = t, v
                if best_v != old:
                    m[k] = best_v
                    changed += 1
                    Q, pu = self._Q(m), self._pu(m)
                else:
                    Q[old] += rvec
                    Q[:, old] += cvec
                    Q[old, old] += self_
            if changed == 0:
                break
        return m, self.total(m)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--num-cluster-shards", type=int, default=3, help="of 20; more = more utts, slower")
    p.add_argument("--num-audio-utts", type=int, default=6000)
    p.add_argument("--num-lm-utts", type=int, default=60000)
    p.add_argument("--objective", choices=["ce", "kl", "both"], default="both")
    p.add_argument("--restarts", type=int, default=6)
    p.add_argument("--lam", type=float, default=5.0, help="weight of the unigram KL term")
    p.add_argument("--basin", action="store_true", help="corrupt-the-oracle basin sweep")
    p.add_argument("--collapse", action="store_true", help="Task C: append the collapse-repeats map B")
    p.add_argument("--lam-z", type=float, default=0.0, help="--collapse: weight of (Z - z_target)^2")
    p.add_argument("--z-target", type=float, default=None,
                   help="--collapse: target for Z; default = label-free min(1, E[S]/E[T])")
    p.add_argument("--cache", default="/var/tmp/cheat_seg_identifiability_cache.pkl")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    t0 = time.time()
    clus, phon = load_data(args.num_clusters, args.num_cluster_shards, args.cache)
    common = sorted(set(clus) & set(phon))
    eq = [t for t in common if len(clus[t]) == len(phon[t])]
    if not eq:
        sys.exit("no equal-length utterances -- wrong HDFs?")
    J = oracle_table(clus, phon, eq, args.num_clusters)
    cf, Pph, oracle_acc = report_alignment(clus, phon, common, eq, J, args.num_clusters)
    omap = J.argmax(1)

    def acc(m):
        return float((cf * Pph[np.arange(args.num_clusters), m]).sum())

    # unpaired split: LM text from utterances NOT used on the audio side
    audio_tags = eq[: args.num_audio_utts]
    aset = set(audio_tags)
    lm_tags = [t for t in common if t not in aset][: args.num_lm_utts]
    TBc = np.zeros((NUM_PHON, NUM_PHON))
    TUc = np.zeros(NUM_PHON)
    for t in lm_tags:
        q = phon[t].astype(np.int64)
        np.add.at(TUc, q, 1)
        np.add.at(TBc, (q[:-1], q[1:]), 1)
    TB, TU = TBc / TBc.sum(), TUc / TUc.sum()
    Cb = np.zeros((args.num_clusters, args.num_clusters))
    cu = np.zeros(args.num_clusters)
    for t in audio_tags:
        s = clus[t].astype(np.int64)
        np.add.at(cu, s, 1)
        np.add.at(Cb, (s[:-1], s[1:]), 1)
    Cb /= Cb.sum()
    cu /= cu.sum()

    z_target, bigram_ratio = 1.0, 1.0
    if args.collapse:
        mean_s = float(np.mean([len(phon[t]) for t in lm_tags]))
        mean_t = float(np.mean([len(clus[t]) for t in audio_tags]))
        n_tok = sum(len(clus[t]) for t in audio_tags)
        bigram_ratio = sum(max(len(clus[t]) - 1, 0) for t in audio_tags) / n_tok
        z_target = min(1.0, mean_s / mean_t) if args.z_target is None else args.z_target
        print("\n  collapse (Task C): E[S] %.2f / E[T] %.2f = %.4f -> z_target %.4f, lam_z %.3g"
              % (mean_s, mean_t, mean_s / mean_t, z_target, args.lam_z))

    objectives = ["ce", "kl"] if args.objective == "both" else [args.objective]
    for obj in objectives:
        print(
            "\n=== 4. recover the map from unpaired stats -- objective %r "
            "(%s) ===" % (obj, "LM cross-entropy" if obj == "ce" else "KL(text bigram || induced)")
        )
        srch = Search(Cb, cu, TB, TU, obj, lam=args.lam, collapse=args.collapse, lam_z=args.lam_z,
                      z_target=z_target, bigram_ratio=bigram_ratio)
        rng = np.random.default_rng(args.seed)
        if args.collapse:
            print("  text bigram diagonal backed off: %.5f of the mass dropped + renormalized"
                  % srch.diag_mass)
        print("  oracle map:            loss %.4f  acc %.4f%s"
              % (srch.total(omap), acc(omap), "  Z %.4f" % srch.z_of(omap) if args.collapse else ""))
        best = None
        for r in range(args.restarts):
            m, s = srch.climb(rng.integers(0, NUM_PHON, args.num_clusters), rng)
            print("  restart %d:             loss %.4f  acc %.4f" % (r, s, acc(m)))
            if best is None or s < best[1]:
                best = (m, s)
        print("  best random restart:   loss %.4f  acc %.4f -> PER %.1f%%" % (best[1], acc(best[0]), 100 * (1 - acc(best[0]))))
        m, s = srch.climb(omap, rng)
        print(
            "  climb from the oracle: loss %.4f  acc %.4f (%d/%d assignments moved)%s"
            % (s, acc(m), int((m != omap).sum()), args.num_clusters,
               "  Z %.4f" % srch.z_of(m) if args.collapse else "")
        )
        if s < srch.total(omap) and acc(m) < acc(omap):
            print("  --> the objective moves AWAY from the truth: it ranks a wrong map higher.")
        if best[1] < srch.total(omap):
            print("  --> a random-restart optimum beats the oracle's loss: search alone will not find the truth.")

        if args.basin:
            print("  basin of attraction (start = oracle map corrupted at rate p):")
            for prob in (0.1, 0.3, 0.5, 0.7, 0.9):
                m0 = omap.copy()
                sel = rng.random(args.num_clusters) < prob
                m0[sel] = rng.integers(0, NUM_PHON, int(sel.sum()))
                a0 = acc(m0)
                m, s = srch.climb(m0, rng)
                print("    p=%.1f  start acc %.3f -> final acc %.3f  loss %.4f" % (prob, a0, acc(m), s))

    print("\n(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
