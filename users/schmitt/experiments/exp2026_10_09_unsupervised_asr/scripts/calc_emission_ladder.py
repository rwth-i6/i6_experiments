#!/usr/bin/env python3
"""
Task E.1 -- the supervised-ceiling ladder over audio representations (`PLAN_TASK_E.md`).

Why this exists: II.C.2 measured the *supervised* ceiling of a memoryless map on the real clus128
stream at 71.1% PER, so no unsupervised criterion can produce a usable result there.  The criterion
itself is essentially solved on cheat-seg (26.4% label-free against a 26.0% ceiling, II.D.3).  The
binding constraint is therefore the emission model, and the first thing to do is measure -- WITH
labels -- what ceiling each candidate audio representation admits, before spending any unsupervised
run on it.

Every rung is scored through one fixed harness so the rungs are comparable:

  decode  = argmax phoneme per audio token, collapse adjacent repeats
  metric  = edit distance to the phoneme reference -> PER, on a held-out PAIRED eval set
  fit     = position-proportional alignment -> count map -> +1 Viterbi realignment -> (optional)
            direct PER hill-climb.  Maximum likelihood under the collapse model is NOT used: II.C.2
            showed it decodes to 7 tokens/utt and is the wrong supervised comparator here.

and each rung additionally reports the model-class-free diagnostic

  I(token ; phoneme) = H(phoneme) - H(phoneme | token)  in nats, under the rung's own Viterbi
  alignment -- the generalization of "a clus128 id is worth 0.26 nats".

LABELS ARE USED THROUGHOUT.  E.1 selects a *representation*, not a model; it is a diagnostic, not a
result, and any later label-free number inherits a label-dependent choice of representation.

Rungs (`--rungs`, comma separated; `all` = a..f):
  a  clus128                      control, reproduces II.C.2's 71.1%
  b  pooled/k128                  granularity: 317 -> 159 tokens per utt at the same alphabet size
  c  pooled/k{128,256,512,...}    resolution (`--ks`); k=512 is the analogue of cheat-seg's k512
  d  pooled continuous probe      discretization: linear + MLP on the raw 512-d vectors
  e  context                      log-linear over [c_{t-1}; c_t; c_{t+1}] (and +-2)
  f  segmented                    agglomerative merge to ~one segment per phoneme, pool, re-cluster
  g  segmented, oracle boundaries upper bound: boundaries from a supervised Viterbi alignment

  ./calc_emission_ladder.py --rungs a,b,c --ks 128,512,2048
"""
import argparse
import heapq
import os
import sys
import time

import numpy as np
import torch

from calc_cheat_seg_identifiability import NUM_PHON, _load_hdfs, PHONEME_HDF
from calc_corpus_diagnostics import CLUS128_HDF, read_vocab, VOCAB
from calc_unsegmented_map import (
    collapse,
    edit_distance,
    per,
    per_shuffled,
    proportional_counts,
    viterbi_align_counts,
    per_coordinate_descent,
)

FEAT_ROOT = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_experiments/users/schmitt/experiments/"
    "exp2025_10_02_shared_enc/librispeech/data/audio_preprocessing/"
    "Wav2VecUFeaturizeAudioJob.mkGrrp0YWy8y/output/audio_features"
)
STAGES = {
    "pca512": "precompute_pca512",                             # 514.4 tokens/utt, one per 20 ms frame
    "cls_mean": "precompute_pca512_cls128_mean",               # 317.5, pooled within cls128 runs
    "pooled": "precompute_pca512_cls128_mean_pooled",          # 159.0, + pairwise 2x subsample
}
LBS_KEY = "train-other-960"


# ---------------------------------------------------------------- feature access


class FeatureStore:
    """
    Random access by seq tag into one `prepare_audio.sh` stage.

    Each stage is `{train,valid}.npy` (flat [sum_T, 512] float32) + `.lengths` + `.tsv`; the tsv has
    the corpus root as its first line and then `146197/1061-146197-0004.flac\\t<n_samples>`, which
    maps to the seq tag `train-other-960/1061-146197-0004/1061-146197-0004` (the same reformatting
    `sis_recipe/data/audio.py::_reformat_tsv_seq_tags` does).  The fairseq 1% `valid` split is part
    of the corpus and is indexed alongside `train`.
    """

    def __init__(self, stage, root=FEAT_ROOT):
        self.dir = os.path.join(root, STAGES[stage])
        self.stage = stage
        self.npy = {}
        self.index = {}
        for split in ("train", "valid"):
            npy = os.path.join(self.dir, "%s.npy" % split)
            if not os.path.exists(npy):
                continue
            self.npy[split] = np.load(npy, mmap_mode="r")
            lens = np.loadtxt(os.path.join(self.dir, "%s.lengths" % split), dtype=np.int64)
            lens = np.atleast_1d(lens)
            tags = self._tags(os.path.join(self.dir, "%s.tsv" % split))
            assert len(tags) == len(lens), (split, len(tags), len(lens))
            off = np.concatenate([[0], np.cumsum(lens)])
            for i, t in enumerate(tags):
                self.index[t] = (split, int(off[i]), int(lens[i]))
        self.dim = int(next(iter(self.npy.values())).shape[1])

    @staticmethod
    def _tags(tsv):
        out = []
        with open(tsv) as f:
            f.readline()  # corpus root
            for line in f:
                name = line.split("\t", 1)[0].rsplit("/", 1)[-1].rsplit(".", 1)[0]
                out.append("%s/%s/%s" % (LBS_KEY, name, name))
        return out

    def __contains__(self, tag):
        return tag in self.index

    def get(self, tag):
        split, off, n = self.index[tag]
        return np.asarray(self.npy[split][off : off + n], dtype=np.float32)

    def length(self, tag):
        return self.index[tag][2]

    def sample_frames(self, tags, n_frames, rng):
        """`n_frames` frames drawn uniformly over the concatenation of `tags` (for a k-means fit)."""
        lens = np.array([self.length(t) for t in tags], dtype=np.int64)
        tot = int(lens.sum())
        take = min(n_frames, tot)
        pick = np.sort(rng.choice(tot, size=take, replace=False))
        off = np.concatenate([[0], np.cumsum(lens)])
        out = np.empty((take, self.dim), dtype=np.float32)
        j = 0
        for i, t in enumerate(tags):
            k = np.searchsorted(pick, off[i + 1])
            if k > j:
                out[j:k] = self.get(t)[pick[j:k] - off[i]]
                j = k
        return out


# ---------------------------------------------------------------- streams


def kmeans_fit(X, k, seed, max_iter=100):
    from sklearn.cluster import MiniBatchKMeans

    km = MiniBatchKMeans(n_clusters=k, random_state=seed, n_init=3, max_iter=max_iter,
                         batch_size=4096, max_no_improvement=20)
    km.fit(X)
    return km.cluster_centers_.astype(np.float32)


def assign_stream(store, tags, centroids, get=None):
    """{tag: int16 cluster ids} by nearest centroid (squared-euclidean, expanded)."""
    cn = (centroids ** 2).sum(1)
    out = {}
    for t in tags:
        F = store.get(t) if get is None else get(t)
        d = cn[None, :] - 2.0 * (F @ centroids.T)
        out[t] = d.argmin(1).astype(np.int16)
    return out


def agglomerate(F, n_seg):
    """
    Merge adjacent frames of `F [T, D]` until `n_seg` contiguous segments remain, always merging the
    adjacent pair whose (centroid) squared distance is smallest -- Ward-free single-pass agglomerative
    clustering restricted to the sequence, O(T log T) with a lazy heap.  Returns segment boundaries
    as a list of (start, stop).
    """
    T = len(F)
    if n_seg >= T:
        return [(i, i + 1) for i in range(T)]
    n_seg = max(1, n_seg)
    mean = F.astype(np.float64).copy()
    size = np.ones(T)
    alive = np.ones(T, dtype=bool)
    prv = np.arange(-1, T - 1)
    nxt = np.arange(1, T + 1)
    ver = np.zeros(T, dtype=np.int64)

    def dist(i, j):
        d = mean[i] - mean[j]
        return float(d @ d)

    heap = [(dist(i, i + 1), i, i + 1, 0, 0) for i in range(T - 1)]
    heapq.heapify(heap)
    cur = T
    while cur > n_seg and heap:
        d, i, j, vi, vj = heapq.heappop(heap)
        if not alive[i] or not alive[j] or ver[i] != vi or ver[j] != vj or nxt[i] != j:
            continue
        mean[i] = (mean[i] * size[i] + mean[j] * size[j]) / (size[i] + size[j])
        size[i] += size[j]
        alive[j] = False
        ver[i] += 1
        nxt[i] = nxt[j]
        if nxt[i] < T:
            prv[nxt[i]] = i
            heapq.heappush(heap, (dist(i, nxt[i]), i, nxt[i], ver[i], ver[nxt[i]]))
        if prv[i] >= 0:
            p = prv[i]
            heapq.heappush(heap, (dist(p, i), p, i, ver[p], ver[i]))
        cur -= 1
    bounds = []
    i = 0
    while i < T:
        j = nxt[i]
        bounds.append((i, min(j, T)))
        i = j
    return bounds


def segment_pool(store, tags, ratio, overseg, bounds_of=None):
    """
    Pool each utterance's features over contiguous segments and return {tag: [n_seg, D]}.

    `bounds_of(tag)` supplies the boundaries when given (rung g); otherwise they come from the
    label-free agglomerative merge down to `round(overseg * T / ratio)` segments, with `ratio` the
    unpaired token/phoneme length ratio.
    """
    out = {}
    for t in tags:
        F = store.get(t)
        if bounds_of is not None:
            b = bounds_of(t)
        else:
            b = agglomerate(F, int(round(overseg * len(F) / ratio)))
        out[t] = np.stack([F[s:e].mean(0) for s, e in b]).astype(np.float32)
    return out


# ---------------------------------------------------------------- supervised ceiling


def _mi_nats(J):
    """I(token; phoneme) in nats from a joint count table."""
    P = J / max(J.sum(), 1.0)
    px = P.sum(1, keepdims=True)
    py = P.sum(0, keepdims=True)
    nz = P > 0
    return float((P[nz] * np.log(P[nz] / (px @ py)[nz])).sum())


def discrete_ceiling(name, stream, eval_tags, ceil_tags, phon, n, m, args, log=print):
    """
    The supervised ceiling of a memoryless `n x m` table on a discrete token stream.

    Alignment/counting uses the COLLAPSED reference: the emission model followed by a collapse
    decode literally cannot emit an adjacent repeat, and `viterbi_align_counts` drops any utterance
    with T < S -- which at ~1 token per phoneme would be half the corpus.  Scoring stays against the
    raw reference, exactly as in II.C.2.
    """
    ref_c = {t: collapse(phon[t].astype(np.int64)).astype(np.int16) for t in set(ceil_tags)}
    cands = {}
    J = proportional_counts(ceil_tags, stream, ref_c, n, m)
    cands["proportional alignment"] = J.argmax(1)
    logM = np.log(J / np.maximum(J.sum(1, keepdims=True), 1) + 1e-8)
    Jv = viterbi_align_counts(logM, ceil_tags, stream, ref_c, n, m)
    cands["+1 Viterbi realignment"] = Jv.argmax(1)
    n_skip = sum(len(stream[t]) < len(ref_c[t]) for t in ceil_tags)

    best_name, best_per, best_a, best_lh = None, None, None, None
    for nm, a in cands.items():
        e, lh, _ = per(a, eval_tags, stream, phon)
        log("    %-24s PER %6.2f%%  (hyp %.1f tokens)" % (nm, e, lh))
        if best_per is None or e < best_per:
            best_name, best_per, best_a, best_lh = nm, e, a, lh
    if args.per_cd_sweeps and n <= args.per_cd_max_k:
        fit_tags = ceil_tags[: args.num_per_cd_utts]
        freq = np.bincount(np.concatenate([stream[t].astype(np.int64) for t in fit_tags]), minlength=n)
        order = np.argsort(-freq)[: args.per_cd_top_k]
        cd = per_coordinate_descent(best_a, fit_tags, stream, phon, m, order,
                                    sweeps=args.per_cd_sweeps, log=lambda s: log("      " + s))
        e, lh, _ = per(cd, eval_tags, stream, phon)
        log("    %-24s PER %6.2f%%  (hyp %.1f tokens)" % ("PER hill-climb", e, lh))
        if e < best_per:
            best_name, best_per, best_a, best_lh = "PER hill-climb", e, cd, lh
    return dict(rung=name, n=n, per=best_per, hyp_len=best_lh, via=best_name,
                mi=_mi_nats(Jv), tokens=float(np.mean([len(stream[t]) for t in eval_tags])),
                skipped=n_skip, assign=best_a)


# ---------------------------------------------------------------- continuous probe


def _frame_targets(feats, ref_c, tags, logits_fn=None):
    """Per-frame phoneme targets: proportional alignment, or Viterbi under `logits_fn` if given."""
    X, Y = [], []
    for t in tags:
        F = feats[t]
        c = ref_c[t].astype(np.int64)
        T, S = len(F), len(c)
        if S == 0 or T < S:
            continue
        if logits_fn is None:
            y = c[(np.arange(T) * S) // T]
        else:
            em = torch.log_softmax(logits_fn(torch.as_tensor(F)), dim=1).numpy()[:, c]
            V = np.full(S, -np.inf)
            V[0] = em[0, 0]
            bp = np.zeros((T, S), dtype=bool)
            for i in range(1, T):
                shift = np.concatenate([[-np.inf], V[:-1]])
                take = shift > V
                bp[i] = take
                V = np.where(take, shift, V) + em[i]
            y = np.empty(T, dtype=np.int64)
            si = S - 1
            for i in range(T - 1, -1, -1):
                y[i] = c[si]
                if bp[i, si]:
                    si -= 1
        X.append(F)
        Y.append(y)
    return np.concatenate(X), np.concatenate(Y)


class _Standardize(torch.nn.Module):
    """Per-dimension whitening in front of the probe: the PCA512 features are far from unit scale,
    and without it the linear model starts at a frame CE of ~12 nats and never recovers."""

    def __init__(self, mu, sd, net):
        super().__init__()
        self.register_buffer("mu", torch.as_tensor(mu))
        self.register_buffer("sd", torch.as_tensor(sd))
        self.net = net

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def _fit_probe(X, Y, m, hidden, epochs, lr, batch, seed, log, norm=None):
    torch.manual_seed(seed)
    d = X.shape[1]
    net = (torch.nn.Linear(d, m) if not hidden else
           torch.nn.Sequential(torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, m)))
    if norm is not None:
        net = _Standardize(norm[0], norm[1], net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.as_tensor(X)
    Yt = torch.as_tensor(Y)
    n = len(Xt)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for k in range(0, n, batch):
            idx = perm[k : k + batch]
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(net(Xt[idx]), Yt[idx])
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(idx)
        log("      epoch %2d: frame CE %.4f" % (ep, tot / n))
    return net


def continuous_ceiling(name, feats, eval_tags, ceil_tags, phon, m, args, hidden=0, log=print):
    """
    The supervised ceiling without quantization: a frame classifier on the raw vectors, fitted to
    alignment targets (proportional, then one Viterbi realignment under the fitted model -- NOT
    iterated, since II.C.2 found iterated Viterbi-EM collapses the hypothesis length).
    """
    ref_c = {t: collapse(phon[t].astype(np.int64)).astype(np.int16) for t in set(ceil_tags)}
    best = None
    net = None
    norm = None
    for stage in ("proportional", "+1 Viterbi"):
        fn = None if net is None else (lambda z, _n=net: _n(z).detach())
        with torch.no_grad():
            X, Y = _frame_targets(feats, ref_c, ceil_tags, logits_fn=fn)
        if norm is None:
            mu = X.mean(0)
            norm = (mu, X.std(0) + 1e-3)
        net = _fit_probe(X, Y, m, hidden, args.probe_epochs, args.probe_lr, args.probe_batch,
                         args.seed, log, norm=norm)
        with torch.no_grad():
            assign_stream_ = {t: net(torch.as_tensor(feats[t])).argmax(1).numpy().astype(np.int16)
                              for t in eval_tags}
        e, lh, _ = per(np.arange(m), eval_tags, assign_stream_, phon)  # identity map: already phonemes
        log("    %-24s PER %6.2f%%  (hyp %.1f tokens)" % (stage, e, lh))
        if best is None or e < best[0]:
            best = (e, lh, stage)
    with torch.no_grad():
        Xe, Ye = _frame_targets(feats, ref_c, ceil_tags[: min(len(ceil_tags), 500)],
                                logits_fn=lambda z: net(z).detach())
        ce = float(torch.nn.functional.cross_entropy(net(torch.as_tensor(Xe)), torch.as_tensor(Ye)))
        pri = np.bincount(Ye, minlength=m).astype(np.float64)
        pri /= pri.sum()
        h0 = float(-(pri[pri > 0] * np.log(pri[pri > 0])).sum())
    return dict(rung=name, n=0, per=best[0], hyp_len=best[1], via=best[2], mi=h0 - ce,
                tokens=float(np.mean([len(feats[t]) for t in eval_tags])), skipped=0, assign=None)


# ---------------------------------------------------------------- context rung


def context_ceiling(name, stream, eval_tags, ceil_tags, phon, n, m, width, args, log=print):
    """
    Log-linear emission over a window of one-hots: logits[t] = sum_{o=-w..w} W_o[c_{t+o}].
    Equivalent to a (2w+1) * n * m parameter table and fitted the same way as the probe, so the
    audio side stays a finite symbol sequence (the II.B / II.D machinery still applies to it).
    """
    offs = list(range(-width, width + 1))

    def onehot_feats(t):
        x = stream[t].astype(np.int64)
        T = len(x)
        F = np.zeros((T, len(offs) * n), dtype=np.float32)
        for k, o in enumerate(offs):
            idx = np.clip(np.arange(T) + o, 0, T - 1)
            F[np.arange(T), k * n + x[idx]] = 1.0
        return F

    feats = {t: onehot_feats(t) for t in set(eval_tags) | set(ceil_tags)}
    r = continuous_ceiling(name, feats, eval_tags, ceil_tags, phon, m, args, hidden=0, log=log)
    r["n"] = len(offs) * n
    return r


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rungs", default="a,b,c,d,e,f")
    p.add_argument("--ks", default="128,256,512,1024,2048")
    p.add_argument("--stages", default="pooled",
                   help="comma separated: %s (the ladder is run once per stage)" % ",".join(sorted(STAGES)))
    p.add_argument("--num-audio-utts", type=int, default=20000, help="k-means fit pool (label-free)")
    p.add_argument("--num-text-utts", type=int, default=60000)
    p.add_argument("--num-eval-utts", type=int, default=1000)
    p.add_argument("--num-ceiling-utts", type=int, default=3000)
    p.add_argument("--kmeans-frames", type=int, default=400000)
    p.add_argument("--kmeans-utts", type=int, default=4000)
    p.add_argument("--clus128-shards", type=int, default=10)
    p.add_argument("--oversegment", default="1.25", help="rung f/g: n_seg = x * T / R; comma separated to sweep")
    p.add_argument("--context-width", type=int, default=1)
    p.add_argument("--seg-ks", default="512", help="rung f/g: alphabet sizes on the segmented stream")
    p.add_argument("--probe-epochs", type=int, default=15)
    p.add_argument("--probe-lr", type=float, default=3e-3)
    p.add_argument("--probe-batch", type=int, default=4096)
    p.add_argument("--probe-hidden", type=int, default=512)
    p.add_argument("--per-cd-sweeps", type=int, default=1, help="supervised PER hill-climb; 0 = off")
    p.add_argument("--per-cd-max-k", type=int, default=600, help="skip the hill-climb above this k")
    p.add_argument("--per-cd-top-k", type=int, default=128, help="only the most frequent symbols")
    p.add_argument("--num-per-cd-utts", type=int, default=150)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--save-npz", default=None)
    args = p.parse_args()
    rungs = set(args.rungs.split(",")) if args.rungs != "all" else set("abcdef")
    ks = [int(v) for v in args.ks.split(",")]
    t_start = time.time()

    # ---------------- data + splits (the II.C.2 harness, on the tag universe of the feature dump)
    stages = args.stages.split(",")
    for st in stages:
        assert st in STAGES, st
    store = FeatureStore(stages[0])
    phon = _load_hdfs([PHONEME_HDF % i for i in range(10)])
    common = sorted(set(store.index) & set(phon))
    rng = np.random.default_rng(args.seed)
    common = [common[i] for i in rng.permutation(len(common))]
    a0 = args.num_audio_utts
    t0 = a0 + args.num_text_utts
    e0 = t0 + args.num_eval_utts
    audio_tags, text_tags = common[:a0], common[a0:t0]
    eval_tags, ceil_tags = common[t0:e0], common[e0 : e0 + args.num_ceiling_utts]
    assert eval_tags and ceil_tags, "not enough utterances"

    # phoneme ids -> compact 0..m-1 over the phonemes present in the (unpaired) text split
    t1_full = np.bincount(np.concatenate([phon[t].astype(np.int64) for t in text_tags]),
                          minlength=NUM_PHON)
    keep_p = np.where(t1_full > 0)[0]
    m = len(keep_p)
    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(m)
    need = set(eval_tags) | set(ceil_tags)
    phon = {t: p_pos[phon[t].astype(np.int64)].astype(np.int16) for t in need}
    assert all((phon[t] >= 0).all() for t in eval_tags)
    inv = read_vocab(VOCAB)

    mean_s = float(np.mean([len(phon[t]) for t in eval_tags]))
    print("=== Task E.1: supervised-ceiling ladder ===")
    print("  utts: %d audio(k-means pool) / %d text / %d eval / %d ceiling (disjoint) of %d common"
          % (len(audio_tags), len(text_tags), len(eval_tags), len(ceil_tags), len(common)))
    print("  phonemes kept: %d   most frequent: %s" % (m, inv[int(keep_p[t1_full[keep_p].argmax()])]))
    print("  phoneme reference %.2f tokens/utt" % mean_s)
    print("  stages: %s" % ", ".join(stages))
    print("  LABELS ARE USED THROUGHOUT -- this is a diagnostic, not a result.\n")

    km_pool = audio_tags[: args.kmeans_utts]
    rows = []

    def report(r):
        rows.append(r)
        print("  -> %-30s ceiling %6.2f%% PER   (%s, %.1f tokens/utt, I = %.3f nats)\n"
              % (r["rung"], r["per"], r["via"], r["tokens"], r["mi"]))

    # ---------------- a: clus128 control (stage independent)
    if "a" in rungs:
        print("[a] clus128 control (II.C.2 measured 71.1% on its own split)")
        clus = _load_hdfs([CLUS128_HDF % i for i in range(args.clus128_shards)])
        miss = [t for t in need if t not in clus]
        if miss:
            print("    %d of %d eval/ceiling utts missing from the clus128 shards -- skipped"
                  % (len(miss), len(need)))
        ev = [t for t in eval_tags if t in clus]
        ce = [t for t in ceil_tags if t in clus]
        report(discrete_ceiling("a clus128", clus, ev, ce, phon, 128, m, args))
        del clus

    for stage in stages:
      store = FeatureStore(stage)
      mean_t_feat = float(np.mean([store.length(t) for t in audio_tags]))
      ratio = mean_t_feat / mean_s
      tag_ = lambda r: "%s %s" % (stage, r)
      print("################ stage %s: %.2f tokens/utt, R = %.3f ################\n"
            % (STAGES[stage], mean_t_feat, ratio))

      # ---------------- b/c: k-means on the chosen stage
      if "b" in rungs or "c" in rungs:
          want = sorted(set(([128] if "b" in rungs else []) + (ks if "c" in rungs else [])))
          print("[b/c] k-means on %s (%d frames from %d utts)"
                % (STAGES[stage], args.kmeans_frames, len(km_pool)))
          X = store.sample_frames(km_pool, args.kmeans_frames, rng)
          print("    sampled %s" % (X.shape,))
          for k in want:
              t0k = time.time()
              cen = kmeans_fit(X, k, args.seed)
              stream = assign_stream(store, sorted(need), cen)
              print("    k=%d (%.0fs fit+assign)" % (k, time.time() - t0k))
              report(discrete_ceiling(tag_("%s k%d" % ("b" if k == 128 else "c", k)),
                                      stream, eval_tags, ceil_tags, phon, k, m, args))
              if k == 512:
                  globals()["_k512_centroids"] = cen
          del X

      # ---------------- d: continuous probe
      if "d" in rungs:
          print("[d] continuous probe on the raw %d-d vectors" % store.dim)
          feats = {t: store.get(t) for t in need}
          report(continuous_ceiling(tag_("d linear probe"), feats, eval_tags, ceil_tags, phon, m, args,
                                    hidden=0))
          report(continuous_ceiling(tag_("d MLP probe (%d)" % args.probe_hidden), feats, eval_tags,
                                    ceil_tags, phon, m, args, hidden=args.probe_hidden))
          del feats

      # ---------------- e: context
      if "e" in rungs:
          print("[e] log-linear context emission on k=128 symbols")
          X = store.sample_frames(km_pool, args.kmeans_frames, rng)
          cen = kmeans_fit(X, 128, args.seed)
          stream = assign_stream(store, sorted(need), cen)
          del X
          for w in range(1, args.context_width + 1):
              report(context_ceiling(tag_("e context +-%d" % w), stream, eval_tags, ceil_tags, phon,
                                     128, m, w, args))

      # ---------------- f/g: segmentation
      for rung, oracle in (("f", False), ("g", True)):
          if rung not in rungs:
              continue
          print("[%s] agglomerative segmentation to {%s} x T/R%s"
                % (rung, args.oversegment, " (ORACLE boundaries)" if oracle else ""))
          bounds_of = None
          if oracle:
              print("    oracle boundaries are not implemented yet -- skipped")
              continue
          for ov in [float(v) for v in str(args.oversegment).split(",")]:
              if ov / ratio >= 0.98:
                  print("    x%.2f: R = %.2f, target segment count not below the token count -- skipped"
                        % (ov, ratio))
                  continue
              t0s = time.time()
              pooled = segment_pool(store, sorted(need), ratio, ov, bounds_of)
              ntok = float(np.mean([len(pooled[t]) for t in eval_tags]))
              print("    x%.2f: %.0fs, %.1f segments/utt (reference %.1f phonemes)"
                    % (ov, time.time() - t0s, ntok, mean_s))
              fit_pool = segment_pool(store, km_pool[: min(len(km_pool), 1500)], ratio, ov)
              Xs = np.concatenate([fit_pool[t] for t in fit_pool])
              for k in [int(v) for v in str(args.seg_ks).split(",")]:
                  cen = kmeans_fit(Xs, k, args.seed)
                  stream = assign_stream(store, sorted(need), cen, get=lambda t: pooled[t])
                  report(discrete_ceiling(tag_("%s seg x%.2f k%d" % (rung, ov, k)), stream, eval_tags,
                                          ceil_tags, phon, k, m, args))
              report(continuous_ceiling(tag_("%s seg x%.2f probe" % (rung, ov)), pooled, eval_tags,
                                        ceil_tags, phon, m, args, hidden=0))
              del pooled, fit_pool, Xs

    # ---------------- summary
    print("=== ladder (gate G1: <= 45% PER qualifies for E.2) ===")
    print("  %-30s %8s %9s %9s %9s %8s" % ("rung", "symbols", "tok/utt", "PER %", "hyp len", "I nats"))
    for r in rows:
        print("  %-30s %8s %9.1f %9.2f %9.1f %8.3f%s"
              % (r["rung"], r["n"] or "-", r["tokens"], r["per"], r["hyp_len"], r["mi"],
                 "   <- clears G1" if r["per"] <= 45.0 else ""))
    if rows:
        best = min(rows, key=lambda r: r["per"])
        print("\n  best rung: %s at %.2f%% PER (%s)" % (best["rung"], best["per"], best["via"]))
        print("  reference points: clus128 71.1% (II.C.2), cheat-seg k512 26.0% (II.B)")
    if args.save_npz and rows:
        np.savez(args.save_npz, **{("per_" + r["rung"].replace(" ", "_")): r["per"] for r in rows})
    print("(%.0fs)" % (time.time() - t_start))


if __name__ == "__main__":
    main()
