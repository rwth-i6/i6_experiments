"""Gaussian-emission HSMM decipherment on CONTINUOUS BEST-RQ features (no hard k-means, no GAN).

Motivation (SAE_0.md §0a-REAL): hard-kmeans units cap at ~0.63 PER, insertion-dominant — the
discretization, not the encoder, is the ceiling (a continuous linear probe reaches 0.145). This is the
unsupervised bridge: a hidden semi-Markov model whose hidden states ARE the 39 ARPAbet phones + SIL,
with

  transition  = phone bigram estimated from 𝒯_φ (the §1a LM prior), diagonal removed (explicit
                duration handles staying)  -> FIXED (this is the decipherment side-information),
  emission    = per-phone full-covariance Gaussian on PCA-reduced features (learned),
  duration    = per-phone categorical over 1..Dmax (learned)  -> the HSMM part that fights the
                insertion/over-segmentation that killed the hard-unit track.

Because the transition is a phone-labelled LM, states are anchored to real phone identities (no label
permutation) -> Viterbi state == gold id, PER is direct. Trained by Viterbi (hard) EM with the
transition frozen. Two inits: `oracle` (init Gaussians from gold frames -> model-capability upper
bound) and `kmeans`/GW (unsupervised -> honest number).

CPU only. `--mode synth` validates the machinery on data with a known answer; `--mode real` runs on
the cached frozen-feature dump (see real_repr_probe.py)."""
import argparse
import gzip
import os
import time
from importlib.machinery import SourceFileLoader

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RA = SourceFileLoader("ra_mod", os.path.join(_HERE, "repr_audit.py")).load_module()
NEG = -1e30
TPHI = ("work/i6_experiments/users/wu/experiments/ssl/experiments/sae/phonemize/"
        "PhonemizeCorpusJob.QqW74njzlaWq/output/phonemes.txt.gz")


# --------------------------------------------------------------------------- transition (LM prior)
def phone_bigram_logtrans(tphi, n_lines=200000):
    """40x40 log P(next|cur) from 𝒯_φ with boundary SIL; diagonal zeroed (between-segment only)."""
    P2I, N, sil = RA.PHONE2ID, RA.NUM_PHONES, RA.SIL_ID
    C = np.full((N, N), 1e-3)
    op = gzip.open(tphi, "rt") if tphi.endswith(".gz") else open(tphi, "rt")
    with op as f:
        for i, line in enumerate(f):
            if i >= n_lines:
                break
            seq = [sil] + [P2I[t] for t in line.split() if t in P2I] + [sil]
            for a, b in zip(seq, seq[1:]):
                C[a, b] += 1.0
    np.fill_diagonal(C, 0.0)
    C[C.sum(1) == 0] = 1.0
    return np.log(C / C.sum(1, keepdims=True))


# ------------------------------------------------------------------------------- Gaussian emission
def fit_gaussians(X, lab, N, reg=1e-2):
    """Full-cov Gaussian per state from hard-assigned frames. Returns (mean[N,d], prec[N,d,d],
    half_logdet[N]) with a shared floor for empty/tiny states."""
    d = X.shape[1]
    gmean = X.mean(0)
    gcov = np.cov(X.T) + reg * np.eye(d)
    mean = np.tile(gmean, (N, 1))
    cov = np.tile(gcov, (N, 1, 1))
    for i in range(N):
        m = lab == i
        if m.sum() >= d + 2:
            mean[i] = X[m].mean(0)
            cov[i] = np.cov(X[m].T) + reg * np.eye(d)
    prec = np.linalg.inv(cov)
    half_logdet = 0.5 * np.linalg.slogdet(cov)[1]
    return mean, prec, half_logdet


def emission_ll(X, mean, prec, half_logdet):
    """E[T,N] = log N(x_t; state i). Vectorised Mahalanobis."""
    T, d = X.shape
    N = mean.shape[0]
    const = -0.5 * d * np.log(2 * np.pi)
    E = np.empty((T, N))
    for i in range(N):
        Xc = X - mean[i]
        maha = np.einsum("td,de,te->t", Xc, prec[i], Xc)
        E[:, i] = const - half_logdet[i] - 0.5 * maha
    return E


# ------------------------------------------------------------------------------------ HSMM Viterbi
def hsmm_viterbi(E, logB, logpi, logdur, Dmax):
    """Best phone segmentation. E[T,N] emission ll, logB[N,N] between-segment, logdur[N,Dmax] (d=1..).
    Returns per-frame state path[T]."""
    T, N = E.shape
    Pre = np.zeros((T + 1, N))          # prefix sums of E: Pre[t]=sum_{τ<t} E[τ]
    Pre[1:] = np.cumsum(E, axis=0)
    delta = np.full((T + 1, N), NEG)    # delta[t,i]: best ll, last segment=i ends exactly at t
    M = np.full((T + 1, N), NEG)        # M[s,i]=max_j(delta[s,j]+logB[j,i]); Mj=argmax
    Mj = np.zeros((T + 1, N), np.int32)
    bp_d = np.zeros((T + 1, N), np.int32)
    bp_j = np.zeros((T + 1, N), np.int32)
    for t in range(1, T + 1):
        best = np.full(N, NEG)
        bd = np.zeros(N, np.int32)
        bj = np.zeros(N, np.int32)
        for d in range(1, min(t, Dmax) + 1):
            s = t - d
            base = logpi if s == 0 else M[s]              # [N]
            basej = -np.ones(N, np.int32) if s == 0 else Mj[s]
            cand = logdur[:, d - 1] + (Pre[t] - Pre[s]) + base  # [N]
            upd = cand > best
            best = np.where(upd, cand, best)
            bd = np.where(upd, d, bd)
            bj = np.where(upd, basej, bj)
        delta[t] = best
        bp_d[t] = bd
        bp_j[t] = bj
        # M[t,i] = max_j delta[t,j] + logB[j,i]
        scores = delta[t][:, None] + logB           # [j,i]
        M[t] = scores.max(0)
        Mj[t] = scores.argmax(0)
    path = np.empty(T, np.int32)
    t = T
    i = int(np.argmax(delta[T]))
    while t > 0:
        d = int(bp_d[t, i])
        j = int(bp_j[t, i])
        path[t - d:t] = i
        t -= d
        i = j if j >= 0 else i
    return path


# --------------------------------------------------------------------------------------- duration
def fit_durations(paths, N, Dmax, smooth=0.1):
    """Per-state duration categorical from segment run-lengths (clamped to Dmax)."""
    counts = np.full((N, Dmax), smooth)
    for p in paths:
        i0 = 0
        for k in range(1, len(p) + 1):
            if k == len(p) or p[k] != p[i0]:
                d = min(k - i0, Dmax)
                counts[p[i0], d - 1] += 1
                i0 = k
    return np.log(counts / counts.sum(1, keepdims=True))


# ------------------------------------------------------------------------------------ Viterbi-EM
def viterbi_em(train, logB, logpi, Dmax, init_lab, n_iter=8, reg=1e-2, gold_eval=None, verbose=True,
               prior_scale=1.0):
    """train: list of X[T,d]. init_lab: list of per-frame init state (same shapes). Transition FIXED.
    prior_scale (>1) up-weights the LM/duration prior against the per-frame Gaussian evidence (which
    otherwise dominates, since a segment sums d emission terms vs one transition term)."""
    N = logB.shape[0]
    logB = logB * prior_scale
    logpi = logpi * prior_scale
    Xall = np.concatenate(train)
    lab = np.concatenate(init_lab)
    mean, prec, hld = fit_gaussians(Xall, lab, N, reg)
    logdur = fit_durations(init_lab, N, Dmax) * prior_scale
    for it in range(n_iter):
        paths, off = [], 0
        for X in train:
            E = emission_ll(X, mean, prec, hld)
            paths.append(hsmm_viterbi(E, logB, logpi, logdur, Dmax))
        lab = np.concatenate(paths)
        mean, prec, hld = fit_gaussians(Xall, lab, N, reg)
        logdur = fit_durations(paths, N, Dmax) * prior_scale
        if verbose and gold_eval is not None:
            per = per_of(paths, gold_eval)
            facc = float((lab == np.concatenate(gold_eval)).mean())
            print(f"    iter {it}: train frame_acc={facc:.3f} PER={per:.3f}", flush=True)
    return dict(mean=mean, prec=prec, hld=hld, logdur=logdur, logB=logB, logpi=logpi)


def decode(model, Dmax, Xs):
    """Decode with the (scaled) priors stored in the trained model."""
    return [hsmm_viterbi(emission_ll(X, model["mean"], model["prec"], model["hld"]),
                         model["logB"], model["logpi"], logdur=model["logdur"], Dmax=Dmax) for X in Xs]


def per_of(paths, golds):
    tot_e = tot_l = 0
    for p, g in zip(paths, golds):
        pred = RA.run_length_dedup(np.asarray(p)); pred = pred[pred != RA.SIL_ID]
        ref = RA.gold_phone_tokens(np.asarray(g), drop_sil=True)
        tot_e += RA.levenshtein(pred, ref); tot_l += len(ref)
    return tot_e / tot_l if tot_l else float("nan")


# =============================================================================== synthetic gate
def run_synth(seed=0):
    rng = np.random.default_rng(seed)
    N, d, Dmax = 8, 6, 20
    means = rng.normal(0, 3, (N, d))              # well-separated phones
    A = rng.random((N, N)) + 0.05; np.fill_diagonal(A, 0); A /= A.sum(1, keepdims=True)
    dur_mean = rng.integers(3, 8, N)
    seqs, golds = [], []
    for _ in range(120):
        st = rng.integers(N); frames, g = [], []
        for _ in range(rng.integers(12, 25)):
            dd = max(1, int(rng.poisson(dur_mean[st])))
            frames.append(means[st] + rng.normal(0, 1.0, (dd, d))); g += [st] * dd
            st = rng.choice(N, p=A[st])
        seqs.append(np.concatenate(frames)); golds.append(np.array(g))
    logB = np.log(A + 1e-9); np.fill_diagonal(logB, NEG)
    logpi = np.full(N, -np.log(N))
    # unsupervised-ish init: k-means then align clusters to states by best gold overlap (synth only)
    from sklearn.cluster import KMeans
    Xall = np.concatenate(seqs)
    km = KMeans(N, n_init=5, random_state=0).fit(Xall)
    cl = km.labels_; gl = np.concatenate(golds)
    cmap = {c: np.bincount(gl[cl == c], minlength=N).argmax() for c in range(N)}
    init = [np.array([cmap[c] for c in km.predict(X)]) for X in seqs]
    print(f"[synth] N={N} d={d} utts={len(seqs)}  kmeans-init PER={per_of(init, golds):.3f}", flush=True)
    m = viterbi_em(seqs, logB, logpi, Dmax, init, n_iter=8, gold_eval=golds)
    print(f"[synth] final PER={per_of(decode(m, Dmax, seqs), golds):.3f} "
          f"(expect near 0 if machinery correct)", flush=True)


# =============================================================================== real features
def run_real(layer, n_utts, split, init_mode, pca_dim, Dmax, n_iter, bigram_lines, lm_scale=1.0):
    RRP = SourceFileLoader("rrp", os.path.join(_HERE, "real_repr_probe.py")).load_module()
    utts = RRP.extract_features(layer, n_utts, split, do_vad=False, pool_w=1)
    rng = np.random.RandomState(0)
    idx = rng.permutation(len(utts)); nt = len(utts) // 5
    test_i, train_i = idx[:nt], idx[nt:]
    from sklearn.decomposition import PCA
    Xtr_raw = np.concatenate([utts[i][0] for i in train_i]).astype(np.float32)
    pca = PCA(pca_dim, random_state=0).fit(Xtr_raw)
    def feat(i): return pca.transform(utts[i][0].astype(np.float32))
    train = [feat(i) for i in train_i]; test = [feat(i) for i in test_i]
    gtr = [utts[i][1] for i in train_i]; gte = [utts[i][1] for i in test_i]

    logB = phone_bigram_logtrans(TPHI, bigram_lines)
    logpi = logB[RA.SIL_ID].copy()                       # start near SIL's successors
    logpi = np.where(np.isfinite(logpi), logpi, NEG)
    N = RA.NUM_PHONES
    if init_mode == "oracle":
        init = [g.copy() for g in gtr]                   # init Gaussians from gold frames (upper bound)
    elif init_mode == "kmeans":
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(N, random_state=0, batch_size=4096, n_init=5).fit(np.concatenate(train))
        gl = np.concatenate(gtr); cl = km.labels_
        cmap = np.array([np.bincount(gl[cl == c], minlength=N).argmax() if (cl == c).any() else RA.SIL_ID
                         for c in range(N)])              # NOTE: uses gold overlap -> weak-oracle init
        init = [cmap[km.predict(X)] for X in train]
    else:
        raise ValueError(init_mode)
    print(f"[real] layer{layer+1} pca{pca_dim} Dmax{Dmax} init={init_mode} lm_scale={lm_scale} "
          f"train={len(train)} test={len(test)}  init train_PER={per_of(init, gtr):.3f}", flush=True)
    m = viterbi_em(train, logB, logpi, Dmax, init, n_iter=n_iter, gold_eval=gtr, prior_scale=lm_scale)
    tr = decode(m, Dmax, train); te = decode(m, Dmax, test)
    fte = float((np.concatenate(te) == np.concatenate(gte)).mean())
    print(f"[real] TRAIN PER={per_of(tr, gtr):.3f}   TEST PER={per_of(te, gte):.3f}  "
          f"test frame_acc={fte:.3f}", flush=True)
    print(f"  [ref] hard-kmeans oracle 0.632 | continuous linear probe 0.145", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["synth", "real"], default="synth")
    ap.add_argument("--init", choices=["oracle", "kmeans"], default="oracle")
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--n_utts", type=int, default=500)
    ap.add_argument("--split", default="validation.clean")
    ap.add_argument("--pca", type=int, default=48)
    ap.add_argument("--Dmax", type=int, default=25)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--bigram_lines", type=int, default=200000)
    ap.add_argument("--lm_scale", type=float, default=1.0)
    a = ap.parse_args()
    t0 = time.time()
    if a.mode == "synth":
        run_synth()
    else:
        run_real(a.layer, a.n_utts, a.split, a.init, a.pca, a.Dmax, a.iters, a.bigram_lines, a.lm_scale)
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
