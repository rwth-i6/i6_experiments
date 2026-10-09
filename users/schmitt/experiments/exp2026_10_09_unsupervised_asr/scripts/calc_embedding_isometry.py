#!/usr/bin/env python3
"""
Pre-test for unsupervised dictionary learning (Artetxe et al., ACL 2017 / ACL 2018) between
audio-symbol and phoneme embedding spaces.

The idea under test
-------------------
`vecmap`-style self-learning maps two *independently trained* embedding spaces onto each other
with no dictionary: alternate (1) Procrustes -- W = UV^T from the SVD of X^T D Z, the optimal
orthogonal map for the current dictionary D -- with (2) nearest-neighbour dictionary induction
under X W Z^T. It provably ascends the seed-free global objective

    max_W  sum_i max_j (X_i W) . Z_j        s.t.  W W^T = I

so if the two spaces are approximately *isometric* under the true correspondence, the truth is
a good optimum and can be found without any paired data. This is the exact mechanism Chung et
al. (NeurIPS 2018) used to align speech2vec word embeddings to word2vec text embeddings.

For us: one embedding per audio symbol (cluster id) and one per phoneme, then induce the
cluster->phoneme dictionary. This is aimed at the bottleneck `calc_cheat_seg_identifiability.py`
found -- distribution matching *ranks the truth correctly* but the discrete hill-climb cannot
reach it, i.e. the loss is right and the SEARCH is the problem. Procrustes+NN is a far better
search over the same intra-modality co-occurrence signal.

*The whole approach rests on the isometry assumption, and that is cheap to falsify.* This script
does so before anything gets trained, on the cheating-GMM segmentation (one audio token = one
phoneme), where `calc_cheat_seg_identifiability.py` gives us an oracle 512->41 table to score
against (oracle frame acc 0.740 = 26.0% PER floor; unigram-argmax chance 0.103).

What it measures
----------------
0. EMBEDDINGS. Built independently per modality from co-occurrence only (PPMI + truncated SVD
   over a symmetric window), audio from cluster sequences and text from *disjoint* utterances'
   phoneme sequences -- so the two spaces are unrelated by construction, as vecmap assumes.
   Deliberately NOT the shared denoising encoder's states: that encoder is trained to be
   modality-invariant, so there is no rotation left to recover (W ~ I is already near-optimal
   and we know its implied pairing is wrong -- per-code overlap 0.117, 4.8% frame accuracy),
   and its states are near-degenerate anyway (all cosines ~0.5). Pass `--audio-emb-npz` /
   `--text-emb-npz` to score externally computed per-symbol embeddings (e.g. averaged encoder
   states) through the same harness.

1. ISOMETRY under the oracle correspondence (Mantel test). Aggregate the cluster embeddings to
   phoneme level through the oracle table and correlate the 41x41 audio-side cosine matrix with
   the text-side one; same at cluster level (cos(a_i,a_j) vs cos(t_m(i),t_m(j))). Null = the
   same correlation with the phoneme labels randomly permuted. **A true correlation inside the
   permutation null means the assumption fails and no optimizer can help.**

2. IS THERE AN ORTHOGONAL MAP? Procrustes fitted on the oracle correspondence vs on permuted
   ones, compared by mean paired cosine and by the global objective above. Plus the supervised
   ceiling: fit W on the oracle, then NN-decode -- how well Procrustes+NN can do *knowing the
   answer*, which caps everything the unsupervised loop could reach with these embeddings.

3. THE ACTUAL SELF-LEARNING LOOP, fully unsupervised (Artetxe et al. 2018): similarity-of-
   sorted-similarities initialization, stochastic dictionary induction to escape local optima,
   CSLS retrieval against hubness. Scored by frame accuracy of the induced 512->41 map. NB the
   2017 paper's own §5 reports 0% from a random init, so the 2018 initialization is the part
   that matters; `--seed-pairs N` additionally tries the "25 word dictionary" setting by
   anchoring the N most frequent clusters to their true phoneme.
   Watch `phonemes used` in the output: with only ~41 targets the `sum_i max_j` objective can be
   gamed by collapsing every cluster onto a few popular phonemes (Artetxe's analysis assumes
   vocabulary >> dimensionality, which holds on the audio side but not the text side).

Findings (train-other-960, cheat-seg k512 clusters, 6000 audio utts / 33654 disjoint text utts,
PPMI+SVD embeddings, window 1; oracle acc 0.740 = 26.0% PER floor, unigram chance 0.103)
------------------------------------------------------------------------------------------------
CONTROL -- the embeddings are good. A supervised logistic probe on HELD-OUT clusters reads the
phoneme off an audio-symbol embedding at acc 0.239 / 0.349 / 0.503 / 0.546 for dim 8/16/32/64.
So co-occurrence embeddings carry most of the oracle's 0.740; nothing below is an artifact of
weak embeddings.

1. THE ISOMETRY ASSUMPTION HOLDS, clearly. Phoneme-level Mantel Spearman (audio side aggregated
   through the oracle table) 0.77 / 0.64 / 0.49 / 0.50 at dim 4/8/16/32, against a permutation
   null of 0.00 +- 0.04 -- z +13 to +20, beating 100% of 500 permutations at every setting.
   Cluster level (cross-phoneme pairs only) 0.46 / 0.26 / 0.16 / 0.13, z +7 to +12. Window 1
   dominates 2 and 4 throughout. **The two spaces really do share structure under the truth.**

2. BUT THE vecmap OBJECTIVE DOES NOT RANK THE TRUTH. The seed-free `sum_i max_j (X_i W) . Z_j`
   scores the oracle correspondence at z -1.4 .. +2.2 against the permutation null over every
   dim x window tried; the best case (dim 32, z +2.1) still has 20 of 1000 random permutations
   scoring *higher* than the truth. Mean paired cosine does rank the truth (z +3..+8) but by a hair
   (0.381 vs a null max of 0.380 at dim 32).

3. AND THE HYPOTHESIS CLASS CANNOT EXPRESS THE MAP. Fitting W on the oracle and NN-decoding --
   i.e. the best Procrustes+NN can do *knowing the answer* -- gives acc 0.033 / 0.092 / 0.187 /
   0.356 / 0.455 at dim 4/8/16/32/64 against the oracle's 0.740. An unconstrained linear map
   reaches 0.465 at dim 32 vs orthogonal's 0.356: orthogonality forbids contractions, but 510
   clusters have to land on 40 phonemes, which *is* a contraction. Dropping orthogonality is not
   an option -- it is the only thing stopping the seed-free objective from collapsing everything
   onto one target.

4. SO SELF-LEARNING LANDS BELOW CHANCE: acc 0.007-0.034 at every dim, i.e. under the 0.103
   unigram-argmax baseline, with the 2018 sorted-similarity init performing the same as a random
   init -- consistent with the 2017 paper's own 0% for random initialization. Anchoring the 50
   most frequent clusters to their true phoneme (18.3% of frames, the "25 word dictionary"
   setting) reaches only 0.200, and 0.148 of that is the anchors themselves.

5. THE BIJECTIVE VARIANT DOES NOT RESCUE IT. Grouping the audio embeddings by k-means into 40
   groups and matching them to the 40 phonemes one-to-one (Procrustes + Hungarian) removes the
   contraction problem, and at low dim the objective does rank the truth above the random null
   (z +7.4 / +6.1 / +3.8 / +2.6 at dim 3/4/6/8, ~0 from dim 12 up). It still fails, for two
   independent reasons: stage 1 caps the whole path (the best possible labelling of the k-means
   groups is acc 0.355, and only 0.276 under a bijection -- within-phoneme spread exceeds
   between-phoneme spread), and the alternation finds wrong permutations scoring far *better*
   than the truth (dim 3: obj 0.908 at acc 0.040 vs the truth's 0.623 at 0.151), with a climb
   started at the oracle moving 36 of 40 assignments away from it.

WHY, STRUCTURALLY: identifiability and expressiveness pull the dimensionality in opposite
directions. Low dim starves the orthogonal W of the freedom to fake alignments, so the objective
separates the truth -- but cannot separate 40 targets for retrieval (ceiling 0.03-0.09). High dim
lets Procrustes+NN express the map (ceiling 0.36-0.46) -- but an orthogonal W then has d(d-1)/2
degrees of freedom against only 40*d target constraints and can align nearly any assignment, so
the objective goes flat. Nothing in between satisfies both. vecmap does not face this because
word translation is ~bijective with a vocabulary far larger than the dimensionality (Artetxe's
own §5 states that condition); Chung et al. (NeurIPS 2018) align *word*-level speech and text
embeddings, likewise ~bijective. Our 510 -> 40 many-to-one correspondence breaks it.

Note the failure shape is exactly that of the "ce" objective in
`calc_cheat_seg_identifiability.py`: the truth is beaten by wrong maps and a climb started at the
oracle walks away from it. The "kl" bigram-matching direction there behaves the opposite way (it
ranks the truth correctly and is monotone in accuracy, it just cannot be searched). So the
conclusion is unchanged and reinforced: **keep the distribution-matching objective
(`train_steps/output_stats.py`) and spend the effort on initialization and search.** What test 1
adds is that per-symbol co-occurrence geometry is a legitimate thing to initialize *from* -- it
demonstrably shares structure with the phoneme space and a supervised probe reads 0.55 out of it
-- just not via an orthogonal map fitted to `sum max` similarity.

Findings with AVERAGED ENCODER STATES (`calc_encoder_symbol_embeddings.py`, cheat-seg models at
ep100; `--ext-pca` reduces the 512-d states per modality)
------------------------------------------------------------------------------------------------
Substantially better than co-occurrence embeddings on the parts that were *representational*, and
unchanged on the part that was *search*. Plain denoising baseline (`ReturnnTrainingJob.g2r39Yq1fvEG`):

  ext-pca      probe   isometry rho / z   glob.obj z   ceiling orth   ceiling linear
      8        0.152      0.486 / +13.4      +1.5          0.078          0.079
     16        0.285      0.399 /  +9.4      +2.0          0.158          0.273
     32        0.571      0.410 / +10.1      +5.0          0.469          0.595
     64        0.600      0.450 / +10.9      +7.4          0.533          0.635
    512 (off)     --      0.452 / +11.0      +8.3          0.633          0.740*

* a capacity artifact, not evidence: with 512 dims against 510 source symbols an unconstrained
  linear map can realize *any* lookup table, so hitting the oracle's 0.740 there is trivial. The
  informative columns are the orthogonal ceiling and the low-dim rows.

So two of the three objections raised by the co-occurrence run are largely lifted: the vecmap
objective now **does** rank the truth (z +5.0…+8.3, beating 100% of permutations, vs ~0 before),
and Procrustes+NN can express far more of the map (orthogonal ceiling 0.633 vs 0.455, and at
pca-64 a supervised *linear* map on these per-symbol averages reaches 0.635 = 36.5% PER against
the 0.740 / 26.0% oracle). Per-symbol averaged encoder states are simply a good compact feature.

The third objection stands and is now sharper: **unsupervised self-learning is still at chance**
(acc 0.017-0.049 at every dim, 2018 init no better than random) because the search finds maps that
score *higher than the truth* -- obj 0.4923 vs the oracle's 0.4313 at full dim, 0.5409 vs 0.4688 at
pca-64. The permutation null of test 2 is therefore too weak a test on its own: it only relabels
the true map, while the search ranges over all 40^510 many-to-one maps, and that space contains
better-scoring wrong ones. Anchoring seeds does not bridge it either: 25 anchors (10.1% of frames)
give 0.154, and even 100 anchors (32.2% of frames!) only 0.341.
Restricting the hypothesis space to a **bijection** over 40 k-means groups makes the objective
correct -- the oracle bijection is the single best-scoring permutation (0.9000 vs 0.8921 for the
best random restart) and a fixed point of the alternation (0/40 moved) -- but then stage 1 caps the
path at acc 0.228 and the alternation still never reaches the truth. The trade-off of the previous
section thus reappears in the constraint rather than in the dimensionality.

THE ADVERSARIAL MODEL IS WORSE ON EVERY AXIS. `baseline_lstm-gan` (`amUnjnqOjmRf`) against the
plain baseline: probe 0.335/0.357 (vs 0.571/0.600 at pca 32/64), isometry rho 0.087/0.090/0.076 at
z +2.6…+3.0 (vs 0.41-0.45 at z +10…+11), global-objective z +1.5…+2.0 (vs +5.0…+8.3), orthogonal
ceiling 0.317/0.418/0.532 (vs 0.469/0.533/0.633). I.e. the domain-adversarial loss *degrades* the
cross-modal geometry instead of building it -- it matches the marginals while destroying the
correspondence, exactly the "merged under a permutation" failure documented for the cheat-seg
models. If a per-symbol geometry is ever used as an initialization feature, take it from the plain
denoising model, not the GAN one.

Caveats
-------
- With `--audio-emb-npz`/`--text-emb-npz` always sweep `--ext-pca`: the working dimensionality is
  the single most decisive knob (see both findings sections), and slicing raw encoder dimensions
  would be arbitrary, so reduction goes through a per-modality PCA.
- The two SVDs truncate very asymmetrically (512x512 -> d against 41x41 -> d, the latter nearly
  lossless), which itself perturbs the isometry. Sweep `--dim`.
- Cheat-seg is the easy case: one token = one phoneme, memoryless target map. With the real
  k-means clusters a symbol is a phoneme *fragment* with duration structure and the map becomes
  many-to-one plus durations.

Usage
-----
    ./calc_embedding_isometry.py                       # all three tests
    ./calc_embedding_isometry.py --dim 8 --window 2          # sweep; window 1 is best
    ./calc_embedding_isometry.py --seed-pairs 25       # + weakly supervised variant
    ./calc_embedding_isometry.py --embeddings random   # null control: should fail everything

Plain script, no sisyphus graph load. Shares the HDF cache with
`calc_cheat_seg_identifiability.py` (~25 s cold, seconds warm).
"""

import argparse
import sys
import time

import numpy as np

from calc_cheat_seg_identifiability import NUM_PHON, load_data, oracle_table


# ---------------------------------------------------------------- embeddings


def cooc_counts(seqs, vocab_size, window):
    """Symmetric windowed co-occurrence counts [V, V]."""
    C = np.zeros((vocab_size, vocab_size))
    for s in seqs:
        s = s.astype(np.int64)
        for off in range(1, window + 1):
            if len(s) <= off:
                break
            np.add.at(C, (s[:-off], s[off:]), 1.0)
            np.add.at(C, (s[off:], s[:-off]), 1.0)
    return C


def ppmi_svd(C, dim, cds=0.75, eig=0.5):
    """PPMI (with context distribution smoothing) + truncated SVD -> [V, dim]."""
    total = C.sum()
    if total <= 0:
        sys.exit("empty co-occurrence matrix")
    pw = C.sum(1) / total
    pc = C.sum(0) ** cds
    pc = pc / pc.sum()
    denom = np.outer(pw, pc)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((C / total) / denom)
    pmi[~np.isfinite(pmi)] = 0.0
    ppmi = np.maximum(pmi, 0.0)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    d = min(dim, U.shape[1])
    return U[:, :d] * (S[:d] ** eig)


def unit(E):
    n = np.linalg.norm(E, axis=1, keepdims=True)
    return E / np.maximum(n, 1e-12)


def vecmap_normalize(E):
    """vecmap preprocessing: length-normalize, mean-center, length-normalize again."""
    E = unit(E)
    E = E - E.mean(0, keepdims=True)
    return unit(E)


def whiten(E, eps=1e-6):
    cov = (E.T @ E) / max(E.shape[0], 1)
    w, V = np.linalg.eigh(cov)
    return E @ (V / np.sqrt(np.maximum(w, eps))) @ V.T


# ---------------------------------------------------------------- mapping / search


def procrustes(X, Z, D):
    """Optimal orthogonal X->Z for dictionary D [n, m]: argmax_W Tr(X W Z^T D^T)."""
    U, _, Vt = np.linalg.svd(X.T @ (D @ Z))
    return U @ Vt


def global_objective(X, Z, W):
    """mean_i max_j (X_i W) . Z_j -- the seed-free objective of Artetxe et al. (2017), §5."""
    return float(((X @ W) @ Z.T).max(1).mean())


def csls_scores(X, Z, W, k=10):
    """Cross-domain similarity local scaling: 2 cos - r_target - r_source."""
    S = (X @ W) @ Z.T
    kt = min(k, S.shape[0])
    ks = min(k, S.shape[1])
    r_t = np.sort(S, axis=0)[-kt:].mean(0)  # per target, over sources
    r_s = np.sort(S, axis=1)[:, -ks:].mean(1)  # per source, over targets
    return 2.0 * S - r_t[None, :] - r_s[:, None]


def dict_from_scores(S, keep_prob=1.0, rng=None):
    """Hard source->target assignment, optionally with stochastic dictionary induction."""
    if keep_prob < 1.0:
        mask = rng.random(S.shape) < keep_prob
        mask[np.arange(S.shape[0]), S.argmax(1)] |= mask.sum(1) == 0  # never drop a whole row
        S = np.where(mask, S, -np.inf)
        dead = ~np.isfinite(S).any(1)
        if dead.any():
            S = S.copy()
            S[dead] = 0.0
    return S.argmax(1)


def as_matrix(assign, m):
    D = np.zeros((len(assign), m))
    D[np.arange(len(assign)), assign] = 1.0
    return D


def sorted_sim_scores(X, Z):
    """
    Artetxe et al. (2018) unsupervised initialization, adapted to unequal vocabularies.

    The similarity matrix of an embedding space, with its rows sorted, is invariant to the
    space's own rotation, so the two modalities' sorted rows are directly comparable. vecmap
    requires both matrices to be square and of equal size; ours are 512x512 and 41x41, so each
    sorted row is resampled to a common length by linear interpolation of its quantiles.
    """
    def sqrt_sim(E):
        U, S, _ = np.linalg.svd(E, full_matrices=False)
        return (U * S) @ U.T  # symmetric square root of E E^T

    xs = np.sort(sqrt_sim(X), axis=1)
    zs = np.sort(sqrt_sim(Z), axis=1)
    L = min(xs.shape[1], zs.shape[1])
    grid = np.linspace(0.0, 1.0, L)

    def resample(A):
        src = np.linspace(0.0, 1.0, A.shape[1])
        return np.stack([np.interp(grid, src, row) for row in A])

    xs, zs = vecmap_normalize(resample(xs)), vecmap_normalize(resample(zs))
    return xs @ zs.T


def sorted_sim_init(X, Z):
    return dict_from_scores(sorted_sim_scores(X, Z))


def self_learning(X, Z, init_assign, rng, args, acc, phon_names, label, oracle_obj=None):
    """Artetxe et al. (2017) self-learning loop with (2018) stochastic dictionary induction."""
    m = Z.shape[0]
    assign = init_assign
    keep_prob = args.keep_prob_init
    best_obj, best_assign, stall = -np.inf, assign, 0
    print("  [%s] init: acc %.4f  phonemes used %d/%d" % (label, acc(assign), len(set(assign.tolist())), m))
    for it in range(1, args.iters + 1):
        W = procrustes(X, Z, as_matrix(assign, m))
        obj = global_objective(X, Z, W)
        S = csls_scores(X, Z, W, k=args.csls_k) if args.csls else (X @ W) @ Z.T
        assign = dict_from_scores(S, keep_prob, rng)
        if args.seed_assign is not None:
            assign[args.seed_idx] = args.seed_assign
        if obj > best_obj + 1e-6:
            best_obj, best_assign, stall = obj, assign, 0
        else:
            stall += 1
            if stall >= args.patience:
                if keep_prob >= 1.0:
                    break
                keep_prob, stall = min(1.0, keep_prob * 2.0), 0
        if it % args.log_every == 0 or it == 1:
            print(
                "    iter %3d: obj %.4f  keep_prob %.2f  acc %.4f  phonemes used %d/%d"
                % (it, obj, keep_prob, acc(assign), len(set(assign.tolist())), m)
            )
    W = procrustes(X, Z, as_matrix(best_assign, m))
    a = acc(best_assign)
    print(
        "  [%s] final: obj %.4f  acc %.4f -> PER %.1f%%  phonemes used %d/%d"
        % (label, global_objective(X, Z, W), a, 100 * (1 - a), len(set(best_assign.tolist())), m)
    )
    if oracle_obj is not None:
        found = global_objective(X, Z, W)
        print(
            "    vs the oracle correspondence: obj %+.4f (search) %s %+.4f (truth) -- %s"
            % (
                found,
                ">" if found > oracle_obj else "<=",
                oracle_obj,
                "the search found a BETTER-scoring map than the truth, so the permutation null in\n"
                "      test 2 was too weak: it only relabels the true map, while the search ranges over\n"
                "      all 40^%d many-to-one maps" % len(init_assign)
                if found > oracle_obj
                else "the truth still scores highest; the search is what fails",
            )
        )
    top = np.argsort(-np.bincount(best_assign, minlength=m))[:5]
    print("    most used targets: %s" % ", ".join("%s(%d)" % (phon_names[t], (best_assign == t).sum()) for t in top))
    return best_assign, a


def supervised_probe(X, omap, cfk, m, seed):
    """
    Control: can a SUPERVISED classifier read the phoneme off an audio-symbol embedding?
    Held-out clusters only, frame-mass weighted -- the analogue of the supervised linear probe
    that calibrates `calc_shuffle_control.py`. Separates "the embeddings are weak" from "the
    unsupervised mapping is weak".
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(omap))
    tr, te = idx[: len(idx) // 2], idx[len(idx) // 2 :]
    w = cfk[te] / cfk[te].sum()
    out = {}
    for name, clf in (
        ("logreg", LogisticRegression(max_iter=2000, C=10.0)),
        ("5-nn", KNeighborsClassifier(n_neighbors=5, metric="cosine")),
    ):
        clf.fit(X[tr], omap[tr])
        out[name] = float((w * (clf.predict(X[te]) == omap[te])).sum())
    return out


def linear_map(X, Z, D, ridge=1e-3):
    """Unconstrained least-squares map X->Z for dictionary D (diagnostic only, see below)."""
    Zt = D @ Z
    A = X.T @ X + ridge * np.eye(X.shape[1])
    return np.linalg.solve(A, X.T @ Zt)


def kmeans_groups(X, k, seed):
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X).labels_


def group_mass(labels, k, cfk, Pk):
    """G[g, p] = frame mass of group g whose true phoneme is p."""
    G = np.zeros((k, Pk.shape[1]))
    np.add.at(G, labels, Pk * cfk[:, None])
    return G


def group_centroids(X, labels, k, cfk):
    A = np.zeros((k, X.shape[1]))
    for g in range(k):
        sel = labels == g
        if sel.any():
            w = cfk[sel]
            A[g] = (X[sel] * w[:, None]).sum(0) / w.sum()
    return unit(A)


def bijective_match(A, Z, perm0, iters=200):
    """
    Alternate Procrustes with a *permutation* dictionary (Hungarian instead of nearest
    neighbour). Unlike free NN retrieval, a bijection cannot be gamed by collapsing every
    source onto a few popular targets, so the objective is not maximized by a contraction.
    """
    from scipy.optimize import linear_sum_assignment

    m = Z.shape[0]
    perm = perm0.copy()
    for _ in range(iters):
        W = procrustes(A, Z, as_matrix(perm, m))
        row, col = linear_sum_assignment(-((A @ W) @ Z.T))
        new = perm.copy()
        new[row] = col
        if (new == perm).all():
            break
        perm = new
    W = procrustes(A, Z, as_matrix(perm, m))
    return perm, float(((A @ W) * Z[perm]).sum(1).mean())


# ---------------------------------------------------------------- stats helpers


def offdiag(M):
    n = M.shape[0]
    iu = np.triu_indices(n, 1)
    return M[iu]


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float(ra @ rb / max(np.linalg.norm(ra) * np.linalg.norm(rb), 1e-12))


def pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def report_null(name, true_val, null_vals):
    null_vals = np.asarray(null_vals)
    mu, sd = null_vals.mean(), null_vals.std()
    z = (true_val - mu) / max(sd, 1e-12)
    pct = float((null_vals < true_val).mean())
    print(
        "  %-28s true %+.4f | null %+.4f +- %.4f (max %+.4f)  z %+.1f  beats %.1f%% of permutations"
        % (name, true_val, mu, sd, null_vals.max(), z, 100 * pct)
    )
    return z, pct


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--num-cluster-shards", type=int, default=3, help="of 20; more = more utts, slower")
    p.add_argument("--num-audio-utts", type=int, default=6000)
    p.add_argument("--num-lm-utts", type=int, default=60000)
    p.add_argument("--dim", type=int, default=32, help="embedding dimensionality after SVD")
    p.add_argument("--window", type=int, default=1, help="co-occurrence window (symmetric)")
    p.add_argument("--min-count", type=int, default=20, help="drop clusters rarer than this")
    p.add_argument("--embeddings", choices=["cooc", "random"], default="cooc", help="'random' = null control")
    p.add_argument("--audio-emb-npz", help="npz with 'emb' [num_clusters, F]; overrides --embeddings")
    p.add_argument("--text-emb-npz", help="npz with 'emb' [NUM_PHON, F]; overrides --embeddings")
    p.add_argument("--ext-pca", type=int, default=0,
                   help="reduce external embeddings to this many dims by per-modality PCA (0 = off)")
    p.add_argument("--whiten", action="store_true", help="whiten both spaces after normalization")
    p.add_argument("--permutations", type=int, default=500, help="size of the permutation null")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--keep-prob-init", type=float, default=0.1, help="stochastic dictionary induction")
    p.add_argument("--no-csls", dest="csls", action="store_false", help="plain NN retrieval instead of CSLS")
    p.add_argument("--csls-k", type=int, default=10)
    p.add_argument("--seed-pairs", type=int, default=0, help="anchor the N most frequent clusters (weakly supervised)")
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--num-groups", type=int, default=0, help="test 4: audio groups (0 = one per phoneme)")
    p.add_argument("--skip-self-learning", action="store_true")
    p.add_argument("--skip-group-match", action="store_true")
    p.add_argument("--cache", default="/var/tmp/cheat_seg_identifiability_cache.pkl")
    p.add_argument("--restarts", type=int, default=10, help="test 4 random restarts")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    clus, phon = load_data(args.num_clusters, args.num_cluster_shards, args.cache)
    common = sorted(set(clus) & set(phon))
    eq = [t for t in common if len(clus[t]) == len(phon[t])]
    if not eq:
        sys.exit("no equal-length utterances -- wrong HDFs?")

    # oracle table + accuracy, exactly as in calc_cheat_seg_identifiability.py
    J = oracle_table(clus, phon, eq, args.num_clusters)
    cf = J.sum(1) / J.sum()  # cluster frequency
    Pph = J / np.maximum(J.sum(1, keepdims=True), 1)  # p(phoneme | cluster)
    omap_full = J.argmax(1)
    oracle_acc = float((cf * Pph.max(1)).sum())

    # unpaired split: the text side uses utterances the audio side never sees
    audio_tags = eq[: args.num_audio_utts]
    aset = set(audio_tags)
    lm_tags = [t for t in common if t not in aset][: args.num_lm_utts]

    print("=== 0. data / embeddings ===")
    print("  audio utts %d (equal-length subset), disjoint text utts %d" % (len(audio_tags), len(lm_tags)))
    print("  oracle memoryless map: acc %.4f -> PER floor %.1f%%" % (oracle_acc, 100 * (1 - oracle_acc)))

    # which symbols exist at all
    Ca = cooc_counts([clus[t] for t in audio_tags], args.num_clusters, args.window)
    Ct = cooc_counts([phon[t] for t in lm_tags], NUM_PHON, args.window)
    clus_count = np.array([np.bincount(clus[t].astype(np.int64), minlength=args.num_clusters) for t in audio_tags]).sum(0)
    keep_c = np.where(clus_count >= max(args.min_count, 1))[0]
    keep_p = np.where(Ct.sum(1) > 0)[0]
    print(
        "  clusters kept %d/%d (>= %d occurrences, %.1f%% of frames), phonemes present %d"
        % (len(keep_c), args.num_clusters, args.min_count, 100 * cf[keep_c].sum(), len(keep_p))
    )

    if args.audio_emb_npz or args.text_emb_npz:
        if not (args.audio_emb_npz and args.text_emb_npz):
            sys.exit("--audio-emb-npz and --text-emb-npz must be given together")
        Xf = np.load(args.audio_emb_npz)["emb"]
        Zf = np.load(args.text_emb_npz)["emb"]
        print("  embeddings: external, %s / %s" % (Xf.shape, Zf.shape))
        if args.ext_pca:
            # the two spaces are supposed to be independent, so reduce each on its own; and unlike
            # SVD embeddings their dimensions carry no ordering, so slicing would be arbitrary.
            def pca(E, rows, k):
                mu = E[rows].mean(0, keepdims=True)
                _, _, Vt = np.linalg.svd(E[rows] - mu, full_matrices=False)
                return (E - mu) @ Vt[:k].T

            Xf, Zf = pca(Xf, keep_c, args.ext_pca), pca(Zf, keep_p, args.ext_pca)
            print("  reduced by per-modality PCA to %d dims" % args.ext_pca)
    elif args.embeddings == "random":
        Xf = rng.standard_normal((args.num_clusters, args.dim))
        Zf = rng.standard_normal((NUM_PHON, args.dim))
        print("  embeddings: RANDOM (null control), dim %d" % args.dim)
    else:
        Xf = ppmi_svd(Ca, args.dim)
        Zf = ppmi_svd(Ct, args.dim)
        print("  embeddings: PPMI+SVD, window %d, dim %d/%d (audio/text)" % (args.window, Xf.shape[1], Zf.shape[1]))

    d = min(Xf.shape[1], Zf.shape[1])
    X = vecmap_normalize(Xf[keep_c, :d])
    Z = vecmap_normalize(Zf[keep_p, :d])
    if args.whiten:
        X, Z = vecmap_normalize(whiten(X)), vecmap_normalize(whiten(Z))
    print("  working dim %d | X %s  Z %s" % (d, X.shape, Z.shape))

    # oracle correspondence restricted to the kept symbols, as indices into Z
    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(len(keep_p))
    omap = p_pos[omap_full[keep_c]]
    if (omap < 0).any():
        sys.exit("a kept cluster's oracle phoneme is absent from the text side")
    phon_names = [str(q) for q in keep_p]

    # accuracy over the FULL frame mass: dropped clusters get the best constant fallback
    fallback = int(p_pos[(Pph * cf[:, None]).sum(0).argmax()])
    Pk = Pph[np.ix_(keep_c, keep_p)]
    cfk = cf[keep_c]
    dropped_credit = float(sum(cf[i] * Pph[i, keep_p[fallback]] for i in range(args.num_clusters) if i not in set(keep_c.tolist())))

    def acc(assign):
        return float((cfk * Pk[np.arange(len(assign)), assign]).sum()) + dropped_credit

    print("  reachable ceiling with these kept clusters: acc %.4f" % acc(omap))

    probe = supervised_probe(X, omap, cfk, len(keep_p), args.seed)
    print(
        "  supervised probe on held-out clusters: logreg %.4f  5-nn %.4f  (chance %.4f)"
        % (probe["logreg"], probe["5-nn"], float((Pph * cf[:, None]).sum(0).max()))
    )

    # ---------------- 1. isometry
    print("\n=== 1. isometry under the oracle correspondence (Mantel test) ===")
    Sa = X @ X.T
    St = Z @ Z.T

    # phoneme level: aggregate cluster embeddings through the oracle table, freq-weighted
    Agg = np.zeros((len(keep_p), d))
    for pos in range(len(keep_p)):
        sel = omap == pos
        if sel.any():
            w = cfk[sel]
            Agg[pos] = (X[sel] * w[:, None]).sum(0) / w.sum()
    have = np.linalg.norm(Agg, axis=1) > 0
    Ap = unit(Agg[have])
    Zp = Z[have]
    a_ph, t_ph = offdiag(Ap @ Ap.T), offdiag(Zp @ Zp.T)
    print("  phoneme level (%d x %d, %d pairs):" % (have.sum(), have.sum(), len(a_ph)))
    true_r, true_rho = pearson(a_ph, t_ph), spearman(a_ph, t_ph)
    null_r, null_rho = [], []
    for _ in range(args.permutations):
        pi = rng.permutation(int(have.sum()))
        Zs = Zp[pi]
        b = offdiag(Zs @ Zs.T)
        null_r.append(pearson(a_ph, b))
        null_rho.append(spearman(a_ph, b))
    report_null("Pearson r", true_r, null_r)
    z_ph, pct_ph = report_null("Spearman rho", true_rho, null_rho)

    # cluster level: compare cos(a_i,a_j) with cos(t_m(i), t_m(j))
    a_cl = offdiag(Sa)
    t_cl = offdiag(St[np.ix_(omap, omap)])
    diff = offdiag((omap[:, None] != omap[None, :]).astype(float)) > 0
    print("  cluster level (%d x %d, %d pairs, %d of them cross-phoneme):" % (len(omap), len(omap), len(a_cl), diff.sum()))
    true_rho_cl = spearman(a_cl[diff], t_cl[diff])
    null_rho_cl = []
    for _ in range(args.permutations):
        pi = rng.permutation(len(keep_p))
        b = offdiag(St[np.ix_(pi[omap], pi[omap])])
        null_rho_cl.append(spearman(a_cl[diff], b[diff]))
    z_cl, pct_cl = report_null("Spearman rho (cross-phon)", true_rho_cl, null_rho_cl)
    if z_ph < 2 and z_cl < 2:
        print("  --> the two spaces are NOT isometric under the truth: vecmap-style mapping cannot work here.")

    # ---------------- 2. does an orthogonal map exist?
    print("\n=== 2. is there an orthogonal map? (Procrustes under the oracle vs permuted) ===")
    Do = as_matrix(omap, len(keep_p))
    Wo = procrustes(X, Z, Do)
    paired_cos = float((((X @ Wo) * Z[omap]).sum(1) * cfk).sum() / cfk.sum())
    obj_o = global_objective(X, Z, Wo)
    print("  oracle:      mean paired cos %+.4f  global obj %+.4f" % (paired_cos, obj_o))
    null_cos, null_obj = [], []
    for _ in range(args.permutations):
        pi = rng.permutation(len(keep_p))
        Dp = as_matrix(pi[omap], len(keep_p))
        Wp = procrustes(X, Z, Dp)
        null_cos.append(float((((X @ Wp) * Z[pi[omap]]).sum(1) * cfk).sum() / cfk.sum()))
        null_obj.append(global_objective(X, Z, Wp))
    report_null("mean paired cos", paired_cos, null_cos)
    report_null("global objective", obj_o, null_obj)
    sup_assign = dict_from_scores(csls_scores(X, Z, Wo, k=args.csls_k) if args.csls else (X @ Wo) @ Z.T)
    sup_acc = acc(sup_assign)
    print(
        "  supervised ceiling (fit W on the oracle, then NN-decode): acc %.4f -> PER %.1f%%"
        % (sup_acc, 100 * (1 - sup_acc))
    )
    print("  --> that is the best Procrustes+NN can do with these embeddings even knowing the answer.")
    Wl = linear_map(X, Z, Do)
    lin_assign = dict_from_scores((X @ Wl) @ Z.T)
    lin_acc = acc(lin_assign)
    print(
        "  same but with an UNCONSTRAINED linear map: acc %.4f -> PER %.1f%%"
        % (lin_acc, 100 * (1 - lin_acc))
    )
    print(
        "  --> orthogonality forces a distance-preserving map, but %d clusters have to land on %d\n"
        "      phonemes, which is a contraction. If the unconstrained map is much better, the\n"
        "      constraint -- not the embeddings -- is what caps the approach. (Dropping it is not an\n"
        "      option: without it the seed-free objective is maximized by collapsing everything.)"
        % (len(omap), len(keep_p))
    )

    # ---------------- 3. the actual self-learning loop
    if not args.skip_self_learning:
        print("\n=== 3. unsupervised self-learning (Artetxe et al. 2018 init + stochastic induction) ===")
        args.seed_assign, args.seed_idx = None, None
        init = sorted_sim_init(X, Z)
        self_learning(X, Z, init, rng, args, acc, phon_names, "unsup", oracle_obj=obj_o)
        print("  control: random init (the 2017 paper reports 0% accuracy for this)")
        self_learning(X, Z, rng.integers(0, len(keep_p), len(omap)), rng, args, acc, phon_names, "rand-init")
        if args.seed_pairs:
            order = np.argsort(-cfk)[: args.seed_pairs]
            args.seed_idx, args.seed_assign = order, omap[order]
            print(
                "  weakly supervised: %d most frequent clusters anchored to their true phoneme (%.1f%% of frames)"
                % (args.seed_pairs, 100 * cfk[order].sum() / cfk.sum())
            )
            seeded = rng.integers(0, len(keep_p), len(omap))
            seeded[order] = omap[order]
            self_learning(X, Z, seeded, rng, args, acc, phon_names, "seed-%d" % args.seed_pairs)

    # ---------------- 4. group-then-match: make the correspondence bijective
    if not args.skip_group_match:
        from scipy.optimize import linear_sum_assignment

        k = args.num_groups or len(keep_p)
        print("\n=== 4. group-then-match (%d audio groups <-> %d phonemes, bijective) ===" % (k, len(keep_p)))
        labels = kmeans_groups(X, k, args.seed)
        G = group_mass(labels, k, cfk, Pk)
        row, col = linear_sum_assignment(-G)
        oracle_perm = np.zeros(k, dtype=np.int64)
        oracle_perm[row] = col
        print(
            "  stage 1 ceiling (k-means the audio embeddings, then label the groups):\n"
            "    majority label  acc %.4f -> PER %.1f%%   (free, not a bijection)\n"
            "    best bijection  acc %.4f -> PER %.1f%%"
            % (G.max(1).sum(), 100 * (1 - G.max(1).sum()), G[np.arange(k), oracle_perm].sum(),
               100 * (1 - G[np.arange(k), oracle_perm].sum()))
        )
        A = group_centroids(X, labels, k, cfk)

        def gacc(perm):
            return float(G[np.arange(k), perm].sum())

        Wo2 = procrustes(A, Z, as_matrix(oracle_perm, len(keep_p)))
        obj_true = float(((A @ Wo2) * Z[oracle_perm]).sum(1).mean())
        null = []
        for _ in range(args.permutations):
            pr = rng.permutation(len(keep_p))[:k]
            Wp = procrustes(A, Z, as_matrix(pr, len(keep_p)))
            null.append(float(((A @ Wp) * Z[pr]).sum(1).mean()))
        print("  stage 2 -- does the bijective objective rank the truth?")
        report_null("mean matched cos", obj_true, null)
        best = None
        for r in range(args.restarts):
            pm, ob = bijective_match(A, Z, rng.permutation(len(keep_p))[:k])
            if best is None or ob > best[1]:
                best = (pm, ob)
        print("    best of %d random restarts: obj %+.4f  acc %.4f -> PER %.1f%%"
              % (args.restarts, best[1], gacc(best[0]), 100 * (1 - gacc(best[0]))))
        row, col = linear_sum_assignment(-sorted_sim_scores(A, Z))
        si = np.zeros(k, dtype=np.int64)
        si[row] = col
        pm, ob = bijective_match(A, Z, si)
        print("    from the sorted-similarity init: obj %+.4f  acc %.4f -> PER %.1f%%"
              % (ob, gacc(pm), 100 * (1 - gacc(pm))))
        pm, ob = bijective_match(A, Z, oracle_perm)
        print("    from the oracle bijection:      obj %+.4f  acc %.4f (%d/%d moved)"
              % (ob, gacc(pm), int((pm != oracle_perm).sum()), k))

    print("\n  reference points: oracle acc %.4f | unigram-argmax chance %.4f" % (oracle_acc, float((Pph * cf[:, None]).sum(0).max())))
    print("(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
