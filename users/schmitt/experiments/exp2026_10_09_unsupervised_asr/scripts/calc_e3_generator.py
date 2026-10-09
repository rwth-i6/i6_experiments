#!/usr/bin/env python3
"""
Task E.3 -- a learned generator on the continuous features, trained with the validated n-gram
objective instead of a GAN (`PLAN_TASK_E.md` E.3, `RESULTS.md` II.E.3).

Why: E.2 (II.E.2) showed that on real audio the criterion still IDENTIFIES the map -- a supervised
start scores strictly better than every cold optimum -- but that the discrete emission is too weak
(I(token; phoneme) = 0.73 nats vs ~2.3 for oracle segment clusters): 0/8 cold seeds, and an optimum
displaced 11-16 PER points from the supervised map.  E.1 measured that the *continuous* features
carry much more (supervised linear probe on the x1.25 segmented pooled stream: 32.27% PER,
I = 2.21 nats).  So the one variable changed here is the emission model: a softmax generator over
the 512-d features (wav2vec-U's Conv1d, `--kernel 4`; `--kernel 1` is exactly E.1's linear probe),
and the loss is the order-4 conditional-KL criterion of II.B, `sum_k w_k C_k`, `--w 5,5,10,20`,
back-offs 0.05 / 0.2 -- unchanged, only its induced statistics now come from generator posteriors.

The one real design change (PLAN E.3): q_n can no longer be a contraction of a fixed count table, it
is an average of `p_t (x) p_{t+1} (x) ...` over the whole audio corpus, and a step sees one minibatch.
The loss is evaluated at a RUNNING (EMA) estimate of the corpus statistics and differentiated through
the current batch only:

    Q_n = EMA_n.detach() + (q_n^batch - q_n^batch.detach())

so its value is the corpus-level loss at the running estimate and its gradient is
dL/dQ |_{EMA} . dq^batch/dtheta -- an (up to EMA staleness) unbiased estimate of the full-corpus
gradient, since q^batch is a mean over positions.  The naive alternative, the KL of the batch
estimate itself, is biased (log of a noisy 2.56M-cell estimate) and has no corpus-level value.

Two segmentation arms (the input is always E.1's label-free agglomerative segmentation of the
`pooled` stage, pooled per segment):
  x1.00  R ~ 1.0: tokens ~ phonemes, the plain criterion -- the regime it was validated in.
  x1.25  R ~ 1.26: the better supervised ceiling (32.27 vs 41.67), but repeats are now expected.
         `--no-repeat` drops every n-gram cell with an adjacent repeat on both sides and adds Task C's
         length term `lam_z (Z - z*)^2` (Z = 1 - trace q2, z* = mean phonemes / mean segments, both
         unpaired corpus averages).  For n <= 2 that is exactly Task C's collapse pushforward; for
         n = 3, 4 it is an APPROXIMATION (a window with no repeat is a collapsed n-gram whose interior
         tokens have run length 1, so it is duration-weighted) -- see II.E.3.

Modes:
  features  segment + pool + cache the audio/eval/ceiling sets and the text  (CPU, ~15 min)
  selftest  the loss assembly vs `SoftMapLoss` on a hard map (mandatory gate, CPU)
  ceiling   the supervised generator of this class (LABELS; a reference, never a result)
  train     one label-free run: `--init uniform --seed N`, or `--init ceiling` (the
            identifiability probe -- starts from the supervised generator, so it uses labels)

Nothing uses labels outside `ceiling`, `--init ceiling`, and the after-the-fact PER/shuffle scoring;
seed selection and keep-best use the HARDENED loss only.

  ./calc_e3_generator.py features --overseg 1.25
  ./calc_e3_generator.py selftest
  ./calc_e3_generator.py ceiling  --overseg 1.25 --kernel 4
  ./calc_e3_generator.py train    --overseg 1.25 --kernel 4 --no-repeat --init uniform --seed 1
"""
import argparse
import glob
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from calc_unsegmented_map import collapse, per, per_shuffled

DEFAULT_DIR = "/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/e3"
E2_STREAM = ("/work/asr4/schmitt/experiments/2026_04_09_unsupervised_asr/plan_runs/taskE/e2/"
             "pooled_seg1.00_k512/stream.npz")
EPS = 1e-9  # == calc_soft_map_search.EPS, so the self-test compares like with like


# ---------------------------------------------------------------- features (CPU)

_STORE = None


def _pool_one(args):
    tag, ratio, overseg = args
    from calc_emission_ladder import agglomerate

    Fx = _STORE.get(tag)
    b = agglomerate(Fx, int(round(overseg * len(Fx) / ratio)))
    return np.stack([Fx[s:e].mean(0) for s, e in b]).astype(np.float32)


def build_features(args):
    """Same splits + seed as E.1/E.2 (`calc_e2_unsup_map.build_stream`), so the eval set is theirs."""
    global _STORE
    from calc_cheat_seg_identifiability import NUM_PHON, _load_hdfs, PHONEME_HDF
    from calc_emission_ladder import FeatureStore

    _STORE = store = FeatureStore(args.stage)
    phon = _load_hdfs([PHONEME_HDF % i for i in range(10)])
    common = sorted(set(store.index) & set(phon))
    rng = np.random.default_rng(args.split_seed)
    common = [common[i] for i in rng.permutation(len(common))]
    a0 = args.num_split_audio
    t0 = a0 + args.num_split_text
    e0 = t0 + args.num_eval_utts
    audio_pool, text_pool = common[:a0], common[a0:t0]
    eval_tags, ceil_tags = common[t0:e0], common[e0 : e0 + args.num_ceiling_utts]
    audio_tags = audio_pool[: args.num_audio_utts]
    lm_tags = text_pool[: args.num_lm_utts]

    t1_full = np.bincount(np.concatenate([phon[t].astype(np.int64) for t in text_pool]), minlength=NUM_PHON)
    keep_p = np.where(t1_full > 0)[0]
    m = len(keep_p)
    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(m)
    phon_c = {t: p_pos[phon[t].astype(np.int64)].astype(np.int16)
              for t in set(text_pool) | set(eval_tags) | set(ceil_tags)}
    assert all((phon_c[t] >= 0).all() for t in eval_tags)

    # both unpaired: reference length from the TEXT split, feature length from the AUDIO split
    mean_s = float(np.mean([len(phon_c[t]) for t in text_pool[:20000]]))
    mean_t_feat = float(np.mean([store.length(t) for t in audio_pool[:4000]]))
    ratio = mean_t_feat / mean_s
    print("=== E.3 features: stage %s, overseg %.2f ===" % (args.stage, args.overseg))
    print("  feature stage %.2f tokens/utt, text %.2f phonemes/utt (unpaired) -> R = %.3f"
          % (mean_t_feat, mean_s, ratio))

    t0s = time.time()
    out = dict(m=m, keep_p=keep_p, stage=args.stage, overseg=args.overseg, ratio=ratio,
               mean_text_len=mean_s)
    with mp.Pool(args.workers) as pool:
        for name, tags in (("audio", audio_tags), ("eval", eval_tags), ("ceil", ceil_tags)):
            feats = pool.map(_pool_one, [(t, ratio, args.overseg) for t in tags], chunksize=16)
            bad = [i for i, f in enumerate(feats) if not np.isfinite(f).all() or np.abs(f).max() > 1e5]
            if name == "audio":
                # the filter `DumpNumpyFeaturesToHdfJobV2(max_abs_value=1e5)` applies for the wav2vec-U
                # setup (NaN seqs are skipped there unconditionally); label-free, training set only.
                # One NaN utt would otherwise turn mu/sd and every step into NaN.
                keep = sorted(set(range(len(tags))) - set(bad))
                feats, tags = [feats[i] for i in keep], [tags[i] for i in keep]
                print("  audio: dropped %d utts with non-finite or |x| > 1e5 features" % len(bad))
            else:
                assert not bad, "%d bad utts in the paired %s set" % (len(bad), name)
            out["%s_lens" % name] = np.array([len(f) for f in feats], dtype=np.int32)
            out["%s_feats" % name] = np.concatenate(feats)
            out["%s_tags" % name] = np.array(tags)
            print("  %-5s %6d utts, %.2f segments/utt  (%.0fs)"
                  % (name, len(tags), float(out["%s_lens" % name].mean()), time.time() - t0s))
    for name, tags in (("lm", lm_tags), ("eval_p", eval_tags), ("ceil_p", ceil_tags)):
        out["%s_lens" % name] = np.array([len(phon_c[t]) for t in tags], dtype=np.int32)
        out["%s_flat" % name] = np.concatenate([phon_c[t] for t in tags]).astype(np.int16)
    X = out["audio_feats"].astype(np.float32)
    out["mu"] = X.mean(0)
    out["sd"] = X.std(0) + 1e-3  # label-free: the unpaired audio set only
    os.makedirs(os.path.dirname(args.feat_npz), exist_ok=True)
    np.savez(args.feat_npz, **out)
    print("  wrote %s (%.1f GB)" % (args.feat_npz, os.path.getsize(args.feat_npz) / 1e9))


class Data:
    """The cached features, as padded-on-demand GPU tensors."""

    def __init__(self, path, device):
        z = np.load(path, allow_pickle=False)
        self.m = int(z["m"])
        self.ratio = float(z["ratio"])
        self.overseg = float(z["overseg"])
        self.mean_text_len = float(z["mean_text_len"])
        self.mu = torch.as_tensor(z["mu"])
        self.sd = torch.as_tensor(z["sd"])
        self.device = device
        self.sets = {}
        for name in ("audio", "eval", "ceil"):
            lens = z["%s_lens" % name].astype(np.int64)
            X = torch.as_tensor(z["%s_feats" % name])  # host memory: float32 does not fit an 11 GB GPU
            self.sets[name] = (X, lens,
                               np.concatenate([[0], np.cumsum(lens)]), [str(t) for t in z["%s_tags" % name]])
        self.phon = {}
        for name in ("lm", "eval_p", "ceil_p"):
            lens = z["%s_lens" % name]
            flat = z["%s_flat" % name]
            off = np.concatenate([[0], np.cumsum(lens)])
            self.phon[name] = [flat[off[i] : off[i + 1]].astype(np.int64) for i in range(len(lens))]

    def batch(self, name, idx):
        """([B, T, D] float32 padded with zeros, [B] lengths) for utterance indices `idx`."""
        X, lens, off, _ = self.sets[name]
        L = torch.as_tensor(lens[idx])
        Tm = int(L.max())
        pos = torch.as_tensor(off[idx])[:, None] + torch.arange(Tm)[None, :]
        valid = torch.arange(Tm)[None, :] < L[:, None]
        pos = torch.where(valid, pos, torch.zeros_like(pos))
        xb = (X[pos] * valid[..., None]).to(self.device)
        return xb, L.to(self.device)

    def num(self, name):
        return len(self.sets[name][1])


# ---------------------------------------------------------------- the generator


class Generator(torch.nn.Module):
    """Standardize -> dropout -> Conv1d(D, m, kernel) over segments.  Kernel 1 = E.1's linear probe;
    kernel 4 = wav2vec-U's generator (padding 2 left / 1 right, so the output is frame-synchronous)."""

    def __init__(self, mu, sd, m, kernel, bias=True, dropout=0.0):
        super().__init__()
        self.register_buffer("mu", mu.clone().float())
        self.register_buffer("sd", sd.clone().float())
        self.kernel = kernel
        self.drop = torch.nn.Dropout(dropout)
        self.conv = torch.nn.Conv1d(len(mu), m, kernel_size=kernel, bias=bias)

    def forward(self, x, lens):
        valid = (torch.arange(x.shape[1], device=x.device)[None, :] < lens[:, None])[..., None]
        x = ((x - self.mu) / self.sd) * valid  # padding stays exactly 0 after standardization
        x = self.drop(x).transpose(1, 2)
        x = F.pad(x, (self.kernel // 2, self.kernel - 1 - self.kernel // 2))
        return self.conv(x).transpose(1, 2)  # [B, T, m]


def decode(gen, data, name, bs=200):
    """{tag: argmax phoneme per segment} (int16), no collapse (per() collapses)."""
    out = {}
    tags = data.sets[name][3]
    gen.eval()
    with torch.no_grad():
        for s in range(0, len(tags), bs):
            idx = np.arange(s, min(s + bs, len(tags)))
            xb, L = data.batch(name, idx)
            a = gen(xb, L).argmax(-1).cpu().numpy()
            for j, i in enumerate(idx):
                out[tags[i]] = a[j, : int(L[j])].astype(np.int16)
    gen.train()
    return out


def score(gen, data, label, extra=""):
    hyp = decode(gen, data, "eval")
    tags = data.sets["eval"][3]
    ref = dict(zip(tags, data.phon["eval_p"]))
    ident = np.arange(data.m)
    e, lh, lr = per(ident, tags, hyp, ref)
    sh = per_shuffled(ident, tags, hyp, ref)
    print("  %-30s PER %6.2f%%  (hyp %.1f / ref %.1f)   shuffled %6.2f%%  gap %+.2f%s"
          % (label, e, lh, lr, sh, sh - e, extra))
    return dict(per=e, hyp=lh, ref=lr, shuffled=sh, gap=sh - e)


# ---------------------------------------------------------------- the objective


def _norep_masks(m, device):
    """Bool cell masks, True where no two ADJACENT symbols are equal (orders 2..4)."""
    a = torch.arange(m, device=device)
    ne = a[:, None] != a[None, :]
    return {2: ne, 3: ne[:, :, None] & ne[None, :, :], 4: ne[:, :, None, None] & ne[None, :, :, None]
            & ne[None, None, :, :]}


def text_targets(seqs, m, no_repeat, device):
    """t1..t4 (dense, float64) from the unpaired text; `no_repeat` drops + renormalizes repeat cells."""
    from calc_soft_map_search import unigram_bigram, trigram_dense, fourgram_dense

    t1, t2 = unigram_bigram(seqs, m)
    t = {1: t1, 2: t2, 3: trigram_dense(seqs, m), 4: fourgram_dense(seqs, m)}
    t = {k: torch.as_tensor(v / v.sum(), dtype=torch.float64, device=device) for k, v in t.items()}
    dropped = {}
    if no_repeat:
        R = _norep_masks(m, device)
        for k in (2, 3, 4):
            dropped[k] = float(t[k][~R[k]].sum())
            t[k] = t[k] * R[k]
            t[k] = t[k] / t[k].sum()
    return t, dropped


class NgramCriterion:
    """
    II.B's `sum_k w_k C_k` over given INDUCED statistics q1..q4 -- the assembly of
    `calc_soft_map_search.SoftMapLoss.terms/total_of` (cond mode) with the count contraction
    replaced by whatever produced the q's.  Verified against it by `selftest`.
    """

    def __init__(self, t, w, b3, b4, no_repeat=False, lam_z=0.0, z_target=1.0):
        self.t = t
        self.w = [float(v) for v in w]
        self.b3, self.b4 = b3, b4
        self.no_repeat = no_repeat
        self.lam_z, self.z_target = lam_z, z_target
        m = t[1].shape[0]
        self.R = _norep_masks(m, t[1].device) if no_repeat else None
        self.tm = {k: t[k].sum(-1) for k in (2, 3, 4)}  # marginal over the LAST symbol

    @staticmethod
    def _kl(t, q):
        nz = t > 0
        return (t[nz] * (t[nz].log() - torch.log(q[nz] + EPS))).sum()

    def _mask(self, q, k):
        if self.R is None:
            return q
        q = q * self.R[k]
        return q / (q.sum() + EPS)

    def __call__(self, q1, q2, q3, q4):
        """Returns (total, dict of terms).  q's are normalized joints, any dtype; assembled in float64."""
        q1, q2, q3, q4 = (x.double() for x in (q1, q2, q3, q4))
        z = torch.zeros((), dtype=torch.float64, device=q1.device)
        Z = 1.0 - torch.diagonal(q2).sum()
        if self.no_repeat:  # Task C's pushforward, exact at n <= 2
            q1 = (q1 - torch.diagonal(q2)).clamp_min(0.0)
            q1 = q1 / (q1.sum() + EPS)
            q2 = self._mask(q2, 2)
            z = self.lam_z * (Z - self.z_target) ** 2
        if self.b3:
            q3 = (1.0 - self.b3) * q3 + self.b3 * q2[:, :, None] * q1[None, None, :]
        q3 = self._mask(q3, 3)
        if self.b4:
            cond = q3[None, :, :, :] / (q2[None, :, :, None] + EPS)
            q4 = (1.0 - self.b4) * q4 + self.b4 * q3[:, :, :, None] * cond
        q4 = self._mask(q4, 4)
        t = self.t
        c1 = self._kl(t[1], q1)
        c2 = self._kl(t[2], q2) - self._kl(self.tm[2], q2.sum(-1))
        c3 = self._kl(t[3], q3) - self._kl(self.tm[3], q3.sum(-1))
        c4 = self._kl(t[4], q4) - self._kl(self.tm[4], q4.sum(-1))
        w1, w2, w3, w4 = self.w
        total = w1 * c1 + w2 * c2 + w3 * c3 + w4 * c4 + z
        return total, dict(c4=c4, c3=c3, c2=c2, c1=c1, Z=Z, zpen=z)


def fmt_terms(d):
    return "cond4 %.4f, cond3 %.4f, cond2 %.4f, cond1 %.4f, Z %.4f, zpen %.4f" % tuple(
        float(d[k]) for k in ("c4", "c3", "c2", "c1", "Z", "zpen"))


def soft_ngrams(P, lens):
    """
    Normalized soft n-gram joints (orders 1..4) of posteriors `P [B, T, m]`, windows inside each
    utterance only: q_n = mean over valid windows of p_t (x) ... (x) p_{t+n-1}.
    """
    B, T, m = P.shape
    ar = torch.arange(T, device=P.device)[None, :]
    out = []
    for n in (1, 2, 3, 4):
        if T < n:
            out.append(torch.full((m,) * n, 1.0 / m ** n, device=P.device))
            continue
        v = (ar[:, : T - n + 1] + n - 1 < lens[:, None]).reshape(-1)  # [B*(T-n+1)]
        Ps = [P[:, i : T - n + 1 + i].reshape(-1, m)[v] for i in range(n)]
        cnt = max(int(v.sum()), 1)
        if n == 1:
            q = Ps[0].sum(0)
        elif n == 2:
            q = Ps[0].t() @ Ps[1]
        elif n == 3:
            q = ((Ps[0][:, :, None] * Ps[1][:, None, :]).reshape(-1, m * m).t() @ Ps[2]).reshape(m, m, m)
        else:
            A = (Ps[0][:, :, None] * Ps[1][:, None, :]).reshape(-1, m * m)
            Bm = (Ps[2][:, :, None] * Ps[3][:, None, :]).reshape(-1, m * m)
            q = (A.t() @ Bm).reshape(m, m, m, m)
        out.append(q / cnt)
    return out


def hard_ngrams(seqs, m, device):
    """Exact normalized n-gram joints (orders 1..4) of integer sequences, windows inside each sequence."""
    flat = torch.as_tensor(np.concatenate(seqs).astype(np.int64), device=device)
    uid = torch.as_tensor(np.repeat(np.arange(len(seqs)), [len(s) for s in seqs]), device=device)
    out = []
    for n in (1, 2, 3, 4):
        N = len(flat) - n + 1
        code = torch.zeros(N, dtype=torch.int64, device=device)
        for i in range(n):
            code = code * m + flat[i : i + N]
        code = code[uid[:N] == uid[n - 1 :]]
        q = torch.bincount(code, minlength=m ** n).double()
        out.append((q / q.sum()).reshape((m,) * n))
    return out


def hardened_loss(gen, data, crit):
    hyp = decode(gen, data, "audio")
    seqs = list(hyp.values())
    total, terms = crit(*hard_ngrams(seqs, data.m, data.device))
    used = len(np.unique(np.concatenate(seqs)))
    return float(total), terms, used


# ---------------------------------------------------------------- modes


def mode_selftest(args):
    """
    Gate: on a HARD map of E.2's discrete stream, (a) `hard_ngrams` of the mapped sequences and
    (b) `soft_ngrams` of their one-hot posteriors both reproduce `SoftMapLoss.hard` (cond mode,
    same back-offs), term by term.  And the EMA surrogate's gradient equals the full-batch one when
    the EMA equals the batch.
    """
    from calc_e2_unsup_map import load_stream
    from calc_soft_map_search import (SoftMapLoss, unigram_bigram, trigram_dense, trigram_sparse,
                                      fourgram_dense, fourgram_sparse)

    d = load_stream(E2_STREAM)
    n, m = d["k"], d["m"]
    audio = d["audio"][:1000]
    lm = d["lm"][:6000]
    c1, C2 = unigram_bigram(audio, n)
    t1, t2 = unigram_bigram(lm, m)
    t3 = trigram_dense(lm, m)
    t4 = fourgram_dense(lm, m)
    ref = SoftMapLoss(c1, C2, t1 / t1.sum(), t2 / t2.sum(), lam=5.0, C3=trigram_sparse(audio, n), t3=t3 / t3.sum(),
                      tri_scale=1.0, tri_backoff=args.tri_backoff, C4=fourgram_sparse(audio, n), t4=t4 / t4.sum(),
                      four_scale=1.0, four_backoff=args.four_backoff, cond_weights=[float(v) for v in args.w.split(",")])
    t, _ = text_targets(lm, m, False, "cpu")
    crit = NgramCriterion(t, args.w.split(","), args.tri_backoff, args.four_backoff)
    rng = np.random.default_rng(0)
    ok = True
    for trial in range(3):
        assign = rng.integers(0, m, n)
        r_total, r4, r3, r2, r1 = ref.hard(assign, m)
        mapped = [assign[s.astype(np.int64)] for s in audio]
        h_total, h = crit(*hard_ngrams(mapped, m, "cpu"))
        Tm = max(len(s) for s in mapped)
        P = torch.zeros(len(mapped), Tm, m, dtype=torch.float64)
        for i, s in enumerate(mapped):
            P[i, torch.arange(len(s)), torch.as_tensor(s)] = 1.0
        L = torch.as_tensor([len(s) for s in mapped])
        s_total, s = crit(*soft_ngrams(P, L))
        print("  trial %d  SoftMapLoss %.6f (c4 %.5f c3 %.5f c2 %.5f c1 %.5f)" % (trial, r_total, r4, r3, r2, r1))
        print("           hard_ngrams %.6f (c4 %.5f c3 %.5f c2 %.5f c1 %.5f)" % (
            float(h_total), *(float(h[k]) for k in ("c4", "c3", "c2", "c1"))))
        print("           soft_ngrams %.6f" % float(s_total))
        for v in (float(h_total), float(s_total)):
            ok &= abs(v - r_total) < 1e-6 * max(1.0, abs(r_total))
    # the EMA surrogate: value = loss at the EMA, gradient = loss gradient at the EMA through the batch
    torch.manual_seed(0)
    logits = torch.randn(4, 30, m, dtype=torch.float64, requires_grad=True)
    L = torch.as_tensor([30, 25, 17, 30])
    qs = soft_ngrams(torch.softmax(logits, -1), L)
    g_full = torch.autograd.grad(crit(*qs)[0], logits)[0]
    qs = soft_ngrams(torch.softmax(logits, -1), L)
    ema = [q.detach().clone() for q in qs]
    g_sur = torch.autograd.grad(crit(*[e + (q - q.detach()) for e, q in zip(ema, qs)])[0], logits)[0]
    err = float((g_full - g_sur).abs().max() / g_full.abs().max())
    print("  EMA surrogate gradient == full gradient when EMA == batch: rel err %.2e" % err)
    ok &= err < 1e-10
    print("SELFTEST %s" % ("PASSED" if ok else "FAILED"))
    if not ok:
        raise SystemExit(1)


def _viterbi(em, c):
    """Monotone stay-or-advance alignment of `em [T, S]` (log-probs of each ref position) -> [T] indices."""
    T, S = em.shape
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
    return y


def mode_ceiling(args):
    """
    The supervised generator of THIS class on THIS stream: E.1's `continuous_ceiling` protocol
    (proportional alignment to the collapsed reference -> frame CE -> one Viterbi realignment ->
    frame CE; utts with T < S are skipped, as there; the better of the two on the eval set).
    LABELS USED -- a reference and the `--init ceiling` probe's start, never a result.
    """
    dev = _device()
    data = Data(args.feat_npz, dev)
    m = data.m
    ref_c = [collapse(p) for p in data.phon["ceil_p"]]
    lens = data.sets["ceil"][1]
    use = [i for i in range(len(lens)) if 0 < len(ref_c[i]) <= lens[i]]
    print("=== E.3 supervised generator (LABELS USED): overseg %.2f, kernel %d ===" % (data.overseg, args.kernel))
    print("  %d / %d ceiling utts usable (T >= S)" % (len(use), len(lens)))
    targets = {i: ref_c[i][(np.arange(lens[i]) * len(ref_c[i])) // lens[i]] for i in use}
    best = None
    gen = None
    for stage in ("proportional", "+1 Viterbi"):
        if gen is not None:
            with torch.no_grad():
                for s in range(0, len(use), 200):
                    idx = np.array(use[s : s + 200])
                    xb, L = data.batch("ceil", idx)
                    lp = torch.log_softmax(gen(xb, L), -1).cpu().numpy()
                    for j, i in enumerate(idx):
                        targets[i] = _viterbi(lp[j, : lens[i]][:, ref_c[i]], ref_c[i])
        torch.manual_seed(args.seed)
        gen = Generator(data.mu, data.sd, m, args.kernel, dropout=args.dropout).to(dev)
        opt = torch.optim.Adam(gen.parameters(), lr=args.probe_lr)
        for ep in range(args.probe_epochs):
            perm = np.random.default_rng(ep).permutation(use)
            tot = cnt = 0.0
            for s in range(0, len(perm), args.probe_utts):
                idx = perm[s : s + args.probe_utts]
                xb, L = data.batch("ceil", idx)
                Y = torch.full(xb.shape[:2], -100, dtype=torch.int64)
                for j, i in enumerate(idx):
                    Y[j, : lens[i]] = torch.as_tensor(targets[i])
                Y = Y.to(dev)
                logit = gen(xb, L)
                loss = F.cross_entropy(logit.reshape(-1, m), Y.reshape(-1), ignore_index=-100)
                opt.zero_grad()
                loss.backward()
                opt.step()
                k = int((Y >= 0).sum())
                tot += float(loss.detach()) * k
                cnt += k
            if ep % 5 == 4 or ep == args.probe_epochs - 1:
                print("      epoch %2d: frame CE %.4f" % (ep, tot / cnt))
        r = score(gen, data, stage)
        if best is None or r["per"] < best[0]["per"]:
            best = (r, stage, {k: v.detach().cpu().clone() for k, v in gen.state_dict().items()})
    r, stage, sd = best
    print("  ceiling %.2f%% PER via %s (hyp %.1f tokens/utt, shuffle gap %+.2f)" % (r["per"], stage, r["hyp"], r["gap"]))
    torch.save(dict(state=sd, kernel=args.kernel, **r), _ceiling_path(args))
    print("  wrote %s" % _ceiling_path(args))


def corrupt_generator(gen, kind, frac, seed):
    """
    The basin sweep's start (the analogue of cheat-seg's `--basin`): the supervised generator, damaged.

    `perm`: pick round(frac * m) output phonemes and relabel them among themselves by a random
    derangement -- the emission stays sharp, only the labels of that subset are wrong, which is exactly
    the ambiguity a relabeling-invariant signal cannot see.  frac = 1 relabels every phoneme.
    `mix`:  W <- (1 - frac) W + frac * N, N Gaussian with W's Frobenius norm (bias likewise) -- the
            labels stay where they were but the emission is progressively buried in noise.
    """
    g = np.random.default_rng(1000 + seed)
    W, b = gen.conv.weight.data, gen.conv.bias.data
    m = W.shape[0]
    with torch.no_grad():
        if kind == "perm":
            k = int(round(frac * m))
            sub = g.choice(m, size=k, replace=False)
            dst = sub.copy()
            while k > 1 and (dst == sub).any():  # derangement: every chosen label really moves
                dst = g.permutation(sub)
            src = np.arange(m)
            src[dst] = sub  # output row dst[i] now carries what was row sub[i]
            idx = torch.as_tensor(src, device=W.device)
            W.copy_(W[idx].clone())
            b.copy_(b[idx].clone())
            print("  corrupted: %d / %d output phonemes relabeled (derangement)" % (k, m))
        else:
            for t in (W, b):
                n = torch.as_tensor(g.normal(size=tuple(t.shape)), dtype=t.dtype, device=t.device)
                n = n * (t.norm() / n.norm())
                t.copy_((1.0 - frac) * t + frac * n)
            print("  corrupted: weights mixed with %.2f equal-norm Gaussian noise" % frac)


def mode_train(args):
    dev = _device()
    data = Data(args.feat_npz, dev)
    m = data.m
    t, dropped = text_targets(data.phon["lm"], m, args.no_repeat, dev)
    z_target = min(1.0, data.mean_text_len / float(data.sets["audio"][1].mean()))
    crit = NgramCriterion(t, args.w.split(","), args.tri_backoff, args.four_backoff,
                          no_repeat=args.no_repeat, lam_z=args.lam_z, z_target=z_target)
    init_name = args.init if args.init != "corrupt" else "corrupt-%s%.2f" % (args.corrupt_kind, args.corrupt_frac)
    tag = "%s-%d" % (init_name, args.seed)
    print("=== E.3 train: overseg %.2f, kernel %d, init %s, seed %d, w %s, %s ==="
          % (data.overseg, args.kernel, args.init, args.seed, args.w,
             "no-repeat lam_z %g z* %.4f" % (args.lam_z, z_target) if args.no_repeat else "plain"))
    print("  audio %d utts (%.1f segments/utt), text %d utts; %d phonemes; tau %g -> %g over %d steps, "
          "%d utts/step, lr %g, ema %g"
          % (data.num("audio"), float(data.sets["audio"][1].mean()), len(data.phon["lm"]), m,
             args.tau_start, args.tau_end, args.steps, args.batch_utts, args.lr, args.ema))
    if dropped:
        print("  text repeat mass dropped: %s" % ", ".join("n=%d %.4f" % kv for kv in dropped.items()))

    torch.manual_seed(args.seed)
    gen = Generator(data.mu, data.sd, m, args.kernel, dropout=args.dropout).to(dev)
    if args.init == "uniform":
        with torch.no_grad():  # the barycenter: every output uniform, + seed-dependent symmetry breaking
            gen.conv.weight.normal_(0.0, args.init_noise)
            gen.conv.bias.zero_()
    else:
        ck = torch.load(_ceiling_path(args))
        assert ck["kernel"] == args.kernel
        gen.load_state_dict(ck["state"])
        print("  start = the supervised generator (ceiling %.2f%% PER) -- LABELS USED, a probe" % ck["per"])
        if args.init == "corrupt":
            corrupt_generator(gen, args.corrupt_kind, args.corrupt_frac, args.seed)
    h, ht, used = hardened_loss(gen, data, crit)
    print("  start: hardened %.4f (%s) used %d/%d" % (h, fmt_terms(ht), used, m))
    s0 = score(gen, data, "start (%s)" % init_name) if args.init != "uniform" else None

    opt = torch.optim.Adam(gen.parameters(), lr=args.lr)
    taus = np.geomspace(args.tau_start, args.tau_end, args.steps)
    rng = np.random.default_rng(args.seed)
    ema = None
    best = None
    t0 = time.time()
    for step in range(args.steps):
        tau = float(taus[step])
        idx = rng.choice(data.num("audio"), size=args.batch_utts, replace=False)
        xb, L = data.batch("audio", idx)
        P = torch.softmax(gen(xb, L) / tau, -1)
        qb = soft_ngrams(P, L)
        if ema is None:
            ema = [q.detach().clone() for q in qb]
        total, terms = crit(*[e + (q - q.detach()) for e, q in zip(ema, qb)])
        opt.zero_grad()
        total.backward()
        opt.step()
        with torch.no_grad():
            for e, q in zip(ema, qb):
                e.mul_(args.ema).add_(q.detach(), alpha=1.0 - args.ema)
        last = step == args.steps - 1
        if step % args.eval_every == 0 or last:
            h, ht, used = hardened_loss(gen, data, crit)
            if best is None or h < best["hard"]:
                best = dict(step=step, hard=h, terms=ht, used=used,
                            state={k: v.detach().cpu().clone() for k, v in gen.state_dict().items()})
            r = score(gen, data, "step %d" % step) if args.score_every_eval else None
            print("    %6d  tau %.3f  soft %.4f  hard %.4f  used %2d/%d  max_p %.2f  %s  %.0fs"
                  % (step, tau, float(total), h, used, m,
                     float(P.max(-1).values[torch.arange(P.shape[1], device=dev)[None, :] < L[:, None]].mean()),
                     "PER %.2f gap %+.2f" % (r["per"], r["gap"]) if r else "", time.time() - t0))
        elif step % args.log_every == 0:
            print("    %6d  tau %.3f  soft %.4f (%s)" % (step, tau, float(total), fmt_terms(terms)))

    final = hardened_loss(gen, data, crit)[0]
    if best["hard"] < final:
        print("  [%s] keeping step %d (hardened %.4f) over the final step (%.4f)" % (tag, best["step"], best["hard"], final))
        gen.load_state_dict(best["state"])
    print("  [%s] hardened %.4f (%s) | used %d/%d | %.0fs"
          % (tag, best["hard"] if best["hard"] < final else final, fmt_terms(best["terms"]), best["used"], m,
             time.time() - t0))
    s = score(gen, data, "e3 %s" % tag)
    out = args.out or os.path.join(os.path.dirname(args.feat_npz), "%s_k%d_%s_seed%d.pt" % (
        "norep" if args.no_repeat else "plain", args.kernel, init_name, args.seed))
    torch.save(dict(state=gen.state_dict(), hard=min(best["hard"], final), kernel=args.kernel, **s), out)
    print("  wrote %s" % out)


def mode_summary(args):
    """All finished runs of one arm: every seed, hardened loss + PER, the loss-selected one marked."""
    rows = []
    for f in sorted(glob.glob(os.path.join(os.path.dirname(args.feat_npz), "*.pt"))):
        z = torch.load(f)
        if "hard" in z:
            rows.append((os.path.basename(f), z))
    for name, z in sorted(rows, key=lambda r: r[1]["hard"]):
        print("  %-36s hardened %10.4f  PER %6.2f  gap %+6.2f  hyp %.1f"
              % (name, z["hard"], z["per"], z["gap"], z["hyp"]))


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ceiling_path(args):
    return os.path.join(os.path.dirname(args.feat_npz), "ceiling_k%d.pt" % args.kernel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["features", "selftest", "ceiling", "train", "summary"])
    p.add_argument("--dir", default=DEFAULT_DIR)
    p.add_argument("--feat-npz", default=None)
    p.add_argument("--stage", default="pooled")
    p.add_argument("--overseg", type=float, default=1.25)
    p.add_argument("--workers", type=int, default=16)
    # splits: same harness + seed as E.1 / E.2
    p.add_argument("--split-seed", type=int, default=1)
    p.add_argument("--num-split-audio", type=int, default=20000)
    p.add_argument("--num-split-text", type=int, default=60000)
    p.add_argument("--num-eval-utts", type=int, default=1000)
    p.add_argument("--num-ceiling-utts", type=int, default=3000)
    p.add_argument("--num-audio-utts", type=int, default=20000)
    p.add_argument("--num-lm-utts", type=int, default=30000, help="== E.2, so the targets are identical")
    # generator
    p.add_argument("--kernel", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.0)
    # criterion (II.B / II.D.3 settings)
    p.add_argument("--w", default="5,5,10,20")
    p.add_argument("--tri-backoff", type=float, default=0.05)
    p.add_argument("--four-backoff", type=float, default=0.2)
    p.add_argument("--no-repeat", action="store_true")
    p.add_argument("--lam-z", type=float, default=100.0, help="only with --no-repeat (II.C.2: ~100)")
    # optimization
    p.add_argument("--init", choices=["uniform", "ceiling", "corrupt"], default="uniform",
                   help="corrupt = the supervised generator damaged by --corrupt-kind/--corrupt-frac (labels)")
    p.add_argument("--corrupt-kind", choices=["perm", "mix"], default="perm")
    p.add_argument("--corrupt-frac", type=float, default=0.5)
    p.add_argument("--init-noise", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--steps", type=int, default=6000)
    p.add_argument("--batch-utts", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--ema", type=float, default=0.95)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.02)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--score-every-eval", action="store_true", help="PER at every eval (logging only)")
    # supervised reference
    p.add_argument("--probe-epochs", type=int, default=15)
    p.add_argument("--probe-lr", type=float, default=3e-3)
    p.add_argument("--probe-utts", type=int, default=32)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    if args.feat_npz is None:
        args.feat_npz = os.path.join(args.dir, "%s_seg%.2f" % (args.stage, args.overseg), "features.npz")
    t0 = time.time()
    {"features": build_features, "selftest": mode_selftest, "ceiling": mode_ceiling,
     "train": mode_train, "summary": mode_summary}[args.mode](args)
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
