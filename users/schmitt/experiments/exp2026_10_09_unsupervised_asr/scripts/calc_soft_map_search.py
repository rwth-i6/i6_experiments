#!/usr/bin/env python3
"""
Recover the cheat-seg cluster->phoneme map by GRADIENT DESCENT on a SOFT assignment matrix,
against the unpaired bigram+unigram distribution-matching loss.

Why this and not Procrustes
---------------------------
`calc_embedding_isometry.py` established that the `vecmap` objective is the wrong one here: in the
unconstrained many-to-one search space the search finds maps it scores *above* the truth (obj
0.4923 vs the truth's 0.4313), and the orthogonality constraint that forbids that collapse also
forbids the 510->40 contraction the task actually needs. Dropping orthogonality does not help --
mapping every cluster onto a single phoneme direction makes every nearest-neighbour cosine 1.0, so
total collapse *is* the argmax of that objective.

`calc_cheat_seg_identifiability.py` established the complementary half: the distribution-matching
direction `KL(text bigram || induced bigram)` **ranks the truth correctly and is monotone in
accuracy** (loss 0.073 @ acc 0.66 ... 0.216 @ acc 0.12), and what it lacked was the search -- a hard
coordinate hill-climb over the 512x41 integer table "never *increases* accuracy, it only preserves
whatever it starts with".

So this script keeps that loss and replaces the search: a row-stochastic assignment matrix
`M [num_clusters, num_phon]` optimized by Adam. Everything stays differentiable and needs no
pairing:

    M         [512, 41]    the soft cluster->phoneme assignment; rows sum to 1 (the parameters)
    c1, C2    audio        cluster unigram [512] and JOINT bigram [512,512], normalized, fixed
    t1, t2    text         phoneme unigram [41] and JOINT bigram [41,41], from DISJOINT utts, fixed
    q1 = M^T c1            the phoneme unigram the assignment induces
    q2 = M^T C2 M          the phoneme joint bigram the assignment induces

    loss = KL(t2 || q2) + lam * KL(t1 || q1)        lam = 5.0
         + tri_scale * KL(t3 || q3)                 optional (`--tri-scale`, default 0 = off), with
                                                    q3 = sum_ijk C3[i,j,k] M[i,a] M[j,b] M[k,c]
                                                    from the observed audio triples (sparse), mixed
                                                    with q2(a,b) q1(c) at `--tri-backoff`
         + four_scale * KL(t4 || q4)                optional (`--four-scale`, needs tri_scale > 0),
                                                    q4 likewise from the observed audio 4-grams, mixed
                                                    with the conditional back-off q3(a,b,c) q(d|b,c)
                                                    at `--four-backoff`; ~2 s/step, ~6 GB (see item 8)

Both KLs are FORWARD (target in front of the log): mode-covering / zero-avoiding, so every phoneme
and transition the text uses must be used at roughly the right rate. Do NOT flip either to
`-sum q log t` (an LM score): that equals `KL(q||t) + H(q)`, i.e. matching in the mode-*seeking*
direction plus an explicit reward for making the output less diverse -- measured on this data, it
improves monotonically as the map is destroyed (see the LM-score-vs-matching note in CLAUDE.md).

Bigrams here are JOINT `p(a,b)`, not conditional `p(b|a)`, on both sides. By the KL chain rule
`KL(t2||q2) = KL(t1||q1) + E_{a~t1}[KL(t2(.|a)||q2(.|a))]`, so the joint bigram term already
contains the unigram one and `lam` is really a knob on marginal-vs-transition weighting: the total
is `(1+lam) * marginal mismatch + 1 * transition mismatch`. (Exact up to sequence boundaries: `q1`
counts every position, `q2`'s row-marginal misses each sequence's last symbol, ~0.6% at ~164
frames/utt.)

`M` row-stochastic and `c1`/`C2` normalized makes `q1`/`q2` normalized automatically, so no
renormalization and no leaking gradients. Loss/lam conventions are identical to
`calc_cheat_seg_identifiability.py` (which uses P for the text bigram and Q for the induced one),
so the numbers are directly comparable to its hill-climb.

Collapse is now expensive rather than optimal: a collapsed `M` yields a near-delta phoneme marginal
and `KL(t1 || q1)` blows up. That is the anti-collapse term orthogonality was standing in for.

The relaxation gap
------------------
A *soft* `M` has strictly more freedom than any hard map, so it can match the target statistics as a
mixture that no discrete labelling achieves -- statistically right, semantically empty. Guards, all
on by default or reported:
  * temperature annealing (`--tau-start` -> `--tau-end`) plus an optional row-entropy penalty,
  * **every reported accuracy is that of the hardened `argmax M`, never the soft matrix**, and
  * the hardened map's own loss is printed next to the soft one, so the gap is visible: if the soft
    loss falls while the hard loss and hard accuracy do not, the run is exploiting the relaxation.

Initializations (`--init`)
--------------------------
  random         logits ~ N(0, --init-scale)
  uniform        all-zero logits + tiny noise (maximally soft start)
  kmeans         **the geometry warm start**: k-means the per-symbol encoder embeddings from
                 `--audio-emb-npz` into one group per phoneme and one-hot those groups to arbitrary
                 phonemes. Uses geometry for what it is demonstrably good at (grouping clusters --
                 the probe reads phonemes off these embeddings at 0.60) and leaves the 40-way
                 labelling to the loss. Not label information: which group gets which phoneme is
                 arbitrary.
  oracle         diagnostic: start at the truth. Does the optimizer stay?
  oracle-corrupt diagnostic: start at the truth with `--corrupt` of the rows randomized, to measure
                 the basin -- the question the hard climb answered with "accuracy never increases".

Findings (train-other-960 cheat-seg, 6000 audio utts / 33654 disjoint text utts, 512x40 = 20_480
parameters, Adam, 1500-3000 steps, tau 1.0 -> 0.02; seconds per run)
------------------------------------------------------------------------------------------------
Consistency check first: the oracle map's hardened loss here is **0.2125** against
`calc_cheat_seg_identifiability.py`'s 0.213, so the two searches are optimizing the same thing.

1. THE LOSS DISCRIMINATES CORRECTLY -- unlike the vecmap objective. All 20 cold-start optima have
   hardened loss **0.242-0.428, every one WORSE than the oracle's 0.2125**. Contrast the Procrustes
   objective, where the search found maps scoring *better* than the truth (0.4923 vs 0.4313). The
   distribution-matching direction really is the right one, now confirmed in the continuous setting.

2. BUT A COLD START STILL LANDS AT CHANCE. Best of 20 random restarts acc **0.085** (most 0.02-0.08)
   against the 0.103 unigram baseline; the geometry warm start (k-means on the per-symbol encoder
   embeddings, `--init kmeans`) gives 0.067. Loss-based selection among cold starts does not help --
   the lowest-loss restart is not the most accurate one, and the whole range is at chance anyway.

3. THE NEW CAPABILITY: **the continuous search climbs UPHILL in accuracy, which the discrete
   hill-climb never did.** Basin sweep (`--basin`; start = oracle with a fraction of rows randomized):

     corrupted   init acc   final acc   gain
       0.2         0.583      0.654    +0.071
       0.4         0.467      0.607    +0.139
       0.6         0.291      0.487    +0.197
       0.8         0.166      0.347    +0.181
       0.9         0.101      0.212    +0.111
       1.0         0.024      0.058    +0.034

   The sibling script's verdict on the hard climb was "it never *increases* accuracy, it only
   preserves whatever it starts with". So the soft relaxation is exactly the search improvement that
   was missing -- it just needs a start above roughly acc 0.10 to produce real lift.

4. A WEAK ANCHOR IS NOW ENOUGH (`--seed-sweep`; the N most frequent clusters pinned to their true
   phoneme and held fixed throughout):

     anchored   of frames   anchors only   final acc   gained by the statistics
        10         4.5%        0.033         0.121          +0.088
        25        10.1%        0.074         0.195          +0.120
        50        18.3%        0.136         0.225          +0.088
       100        32.2%        0.242         0.380          +0.139
       200        55.5%        0.414         0.571          +0.157

   "anchors only" is the accuracy the pinned rows already account for, so the last column is what the
   *unpaired* statistics contributed on top of the supervision: **+0.09 to +0.16 at every level.**
   That is the first positive unsupervised contribution measured in this project. It also beats the
   vecmap seeded runs at equal supervision (25 anchors: 0.195 vs 0.154; 100: 0.380 vs 0.341).

5. THE RELAXATION GAP IS REAL AND CONTROLLABLE. At `--tau-end 0.1` the soft loss sits 0.12-0.74
   below the hardened one; `0.02` roughly halves that (-0.135 -> -0.076) at no cost in accuracy,
   which is why it is the default. An entropy penalty does not improve accuracy and hurts above
   ~0.2. **Read the hardened columns only** -- the script prints the gap and warns when it is large.

6. THE LOSS'S OWN OPTIMUM IS DISPLACED FROM THE TRUTH. Started *at* the oracle, the optimizer lowers
   the hardened loss 0.2125 -> ~0.14 while accuracy falls 0.740 -> 0.64 (the sibling script saw the
   same: 0.213 -> 0.073 at acc 0.657). So this loss is a good but imperfect proxy: its minimum sits
   near 0.64 accuracy, ~0.10 short of the memoryless ceiling, and that bounds what *any* amount of
   search on it can deliver. Closing that last 0.10 needs a richer matching term (trigram, or a real
   phoneme LM used in the distribution-matching direction rather than as a score), not more search.
   Related caveat: over-optimizing hurts -- 600 steps gave larger basin gains than 3000 -- and there
   is no unsupervised criterion for when to stop, since the loss keeps falling as accuracy declines.

WHERE THIS LEFT IT (bigram+unigram only): the search problem was solved well enough to be useful,
and the remaining blockers were (a) an initializer that reaches acc ~0.10-0.15 without labels and
(b) a matching term whose optimum sits on the truth rather than 0.10 below it. Both are addressed by
the trigram term below.

7. THE TRIGRAM TERM (`--tri-scale 1`, 2026-09-09) MOVES THE OPTIMUM TO THE TRUTH AND MAKES A
   LABEL-FREE COLD START WORK. Same data, same 3000 steps / tau 1.0 -> 0.02, both arms run in the
   same session so the numbers are directly comparable (the bigram arm reproduces items 3/4 above).
   Text: 4.09M trigram tokens -> 23_861/64_000 nonzero cells (37%); audio: 565k trigram tokens ->
   284_525 distinct cluster triples, kept sparse. 2.2% of the text trigram mass lies on cells the
   ORACLE-mapped audio never produces (too few audio frames), so `--tri-backoff` (default 0.05,
   which minimizes the oracle's own KL3: 0.488 raw -> 0.399) is required for any hard map to have a
   finite loss. Oracle reference: total 0.6116 = trigram 0.3991 + bigram 0.1376 + 5 x unigram 0.0150.

   Basin sweep (start = oracle with a fraction of rows randomized), hardened accuracy:

     corrupted   init acc   bigram+unigram   +trigram
       0.2         0.583        0.620         0.713
       0.4         0.467        0.590         0.714
       0.6         0.291        0.545         0.702
       0.8         0.166        0.365         0.668
       0.9         0.101        0.224         0.355
       1.0         0.024        0.049         0.086

   The loss's optimum moved from acc ~0.62-0.64 to ~0.70-0.71 (29% PER vs the 26% memoryless
   floor): started AT the oracle the run now settles at 0.707 (bigram arm: 0.636). The basin is also
   far wider -- a start at acc 0.17 recovers to 0.67 where the bigram arm stalled at 0.36. The
   optimum is still displaced (the settled hard loss 0.43 < the oracle's 0.61), just much less.

   Seed sweep (N most frequent clusters anchored), final accuracy, bigram -> trigram:
     0: 0.097 -> 0.153 | 10: 0.130 -> 0.141 | 25: 0.203 -> 0.302 | 50: 0.224 -> 0.312 |
     100: 0.405 -> 0.525 | 200: 0.568 -> 0.656.  (+0.01 ... +0.12 at every level.)

   COLD START: the 5 random restarts (logits ~ N(0, 2)) are still at chance (0.02-0.12). But the
   UNIFORM start -- all-zero logits, i.e. the maximally soft barycenter, no label information of any
   kind -- reaches **acc 0.313 / 0.213 / 0.258 / 0.271 over seeds 1-4** (mean 0.26, PER ~74%) at
   `--tri-scale 1`, and **0.343** at `--tri-scale 3`, against 0.101 for the same start under the
   bigram loss and 0.103 unigram chance. This is the first above-chance, fully unsupervised result in
   the project. Why uniform and not random: from the barycenter the anneal lets the loss choose the
   basin (deterministic-annealing style), whereas a scale-2 random init is already committed to one
   of the many wrong basins and the loss only refines it. So `--init uniform` is now the default
   recommendation for an unsupervised run.

   The hardened loss now also RANKS RUNS IN ACCURACY ORDER, which it did not before: basin optima
   0.42-0.44 @ acc 0.67-0.71 < oracle 0.61 @ 0.74 (still the one exception) < uniform 0.77-0.89 @
   0.21-0.31 < random 1.04-1.20 @ chance; among the four uniform seeds the lowest loss is the most
   accurate. That gives an unsupervised model-selection criterion.

   Uniform-start sweep (hardened acc; the seed only sets the 1e-3 symmetry-breaking noise, yet the
   outcome depends on it strongly -- the barycenter is a saddle and trajectories diverge):

     tri-scale         seed 1   seed 2   seed 3   seed 4
       1               0.313    0.213    0.258    0.271
       3               0.343    0.187    0.265
      10               0.414    0.220    0.286
      30               0.446
       3, 6000 steps   0.391
      10, 6000 steps   0.467    0.219
       3, lam 1        0.361

   Trigram weight is monotone on every seed (seed 1: 0.313 -> 0.446 from weight 1 to 30), a slower
   anneal (6000 steps) adds ~0.05, `lam` barely matters. Seed 2 is stuck at ~0.20 whatever the
   setting; seed 1 reaches 0.467 (53% PER). Crucially the HARDENED LOSS RANKS THE SEEDS CORRECTLY at
   every weight (weight 10: 5.74 @ 0.414 < 6.92 @ 0.286 < 7.27 @ 0.220; 6000 steps: 5.53 @ 0.467 <
   7.62 @ 0.219), so "run N uniform seeds, keep the lowest hardened loss" is a legitimate fully
   unsupervised procedure and would have picked 0.467 here. The relaxation gap grows with the weight
   (-0.2 / -0.6 / -1.3 / -3.1 at 1 / 3 / 10 / 30) -- read only the hardened columns.

   Cost: ~0.07 s/step (sparse first contraction + two small einsums), ~7-8 min per 3000-step run on
   3 CPU threads, against ~7 s for the bigram loss.

8. THE 4-GRAM TERM (`--four-scale`, 2026-09-09). Same construction one order up. The first
   contraction (over `l`) is again sparse over the 282_290 observed (i,j,k) prefixes; the second (over
   `k`) scatters outer products into the dense two-cluster-index intermediate `[n*n, m, m]` -- there is
   no way around that stage, every ordering has one -- and the third/fourth are batched matmuls
   (`n^2 m^3` = 17 GFLOP). Kept affordable by running those stages in float32 (the KL's log is taken
   in float64; float32 costs 1.7e-10 in q4), in-place `index_add_` and matmul instead of einsum (the
   out-of-place/einsum version copied the 3.4 GB tensor per chunk: 8.1 s/step, 11.7 GB): **~2.0
   s/step on 6 threads, ~6.3 GB**, i.e. ~100 min per 3000-step run. Verified against a direct 4-gram
   count of the relabeled audio (max abs diff 1.7e-10).
   The 4-gram target is badly undersampled on the audio side: text 4.06M tokens -> 194_557/2.56M
   nonzero cells, audio 558k tokens -> 451_928 distinct cluster 4-grams, and **14.2% of the text
   4-gram mass sits on cells the ORACLE-mapped audio never produces** (trigram: 2.2%). So the back-off
   carries real weight: oracle KL4 = 1.50 raw, 1.10 / 1.03 / 0.96 / 0.92 at back-off 0.05 / 0.1 / 0.2 /
   0.3 -- still falling at 0.3, i.e. the 4-gram evidence is weak relative to its sampling noise at
   6000 audio utts. Runs use 0.2. More audio utts (`--num-audio-utts`, more `--num-cluster-shards`)
   would be the principled fix; it changes the data setting, so all arms would have to be re-run.

   RESULTS (`--tri-scale 10 --four-scale 10 --four-backoff 0.2`, 3000 steps, vs `--tri-scale 10`
   alone under identical settings; 3 runs in parallel took ~6 h wall-clock, ~6 s/step under
   contention):

     start                    trigram-10 only    + 4-gram-10     hardened loss (4-gram runs)
     oracle                      0.7166             0.7265         11.32      (27.4% PER; ceiling 0.740)
     uniform, seed 1             0.4140             0.6559         13.73      (34.4% PER)
     uniform, seed 2             0.2200             0.1164         33.21      (chance)
     uniform, seed 3             0.2864             0.1149         30.04      (chance)
     uniform, seed 4                --              0.3079         26.04

   So despite the undersampling, the 4-gram term (a) moves the loss's optimum to within 0.013 of
   the memoryless ceiling and (b) turns the label-free uniform start into **0.656 / 34.4% PER** on
   seed 1 -- better than the SUPERVISED linear map on per-symbol encoder embeddings (36.5%) and 8
   points off the oracle table. The uniform start reaches the full basin on 1 of 4 seeds (seed 4
   lands in an intermediate basin at 0.31, seeds 2/3 at chance), but the hardened loss orders all
   four correctly (13.7 < 26.0 < 30.0 ~ 33.2 for acc 0.656 > 0.308 > 0.115 ~ 0.116), so "N uniform
   seeds, keep the lowest hardened loss" selects the good one without labels -- at ~4x the compute.
   The trajectories converge early (seed 1 was at 0.649 by step 500 and flat after 1000), so fewer
   steps would do, e.g. run 1000 steps per seed and only finish the best.
   Hyperparameter caveat: the back-off 0.2 was picked from the oracle's own KL4 curve (mild
   leakage), the weights 10/10 by analogy to the trigram sweep, nothing else was tuned.

Usage
-----
    ./calc_soft_map_search.py                                  # random restarts + oracle diagnostics
    ./calc_soft_map_search.py --init kmeans --audio-emb-npz /work/.../<job>.ep100.audio.npz
    ./calc_soft_map_search.py --basin                          # can it climb uphill in accuracy?
    ./calc_soft_map_search.py --seed-sweep                     # how weak an anchor is enough?
    ./calc_soft_map_search.py --init random --restarts 20 --steps 1500
    ./calc_soft_map_search.py --init uniform --tri-scale 3          # the label-free cold start that works
    ./calc_soft_map_search.py --basin --tri-scale 1                 # trigram vs bigram basin comparison

Plain script, no sisyphus graph load; CPU torch. Shares the HDF cache with
`calc_cheat_seg_identifiability.py`.
"""

import argparse
import sys
import time

import numpy as np
import torch

from calc_cheat_seg_identifiability import NUM_PHON, load_data, oracle_table

EPS = 1e-9


# ---------------------------------------------------------------- statistics


def unigram_bigram(seqs, vocab_size):
    """Normalized unigram [V] and bigram [V, V] counts over a list of int sequences."""
    u = np.zeros(vocab_size)
    b = np.zeros((vocab_size, vocab_size))
    for s in seqs:
        s = s.astype(np.int64)
        np.add.at(u, s, 1.0)
        if len(s) > 1:
            np.add.at(b, (s[:-1], s[1:]), 1.0)
    if u.sum() <= 0 or b.sum() <= 0:
        sys.exit("empty statistics -- wrong data?")
    return u / u.sum(), b / b.sum()


def trigram_dense(seqs, vocab_size):
    """Normalized JOINT trigram [V, V, V] over int sequences -- the text side (41^3 = 69k cells)."""
    c = np.zeros((vocab_size,) * 3)
    for s in seqs:
        s = s.astype(np.int64)
        if len(s) > 2:
            np.add.at(c, (s[:-2], s[1:-1], s[2:]), 1.0)
    if c.sum() <= 0:
        sys.exit("empty trigram statistics -- wrong data?")
    return c / c.sum()


def trigram_sparse(seqs, vocab_size):
    """
    Normalized JOINT trigram of the audio side, sparse: `(pair_index = i*V + j, k, weight)` over the
    DISTINCT observed triples. Dense 512^3 float64 would be 1 GB and the contraction 5 GFLOP/step; only
    ~1M frames are observed, so the first contraction over `k` is a sparse matmul with nnz ~ 1M instead.
    """
    codes = []
    for s in seqs:
        s = s.astype(np.int64)
        if len(s) > 2:
            codes.append((s[:-2] * vocab_size + s[1:-1]) * vocab_size + s[2:])
    codes = np.concatenate(codes)
    uniq, cnt = np.unique(codes, return_counts=True)
    return uniq // vocab_size, uniq % vocab_size, cnt / cnt.sum()


def fourgram_dense(seqs, vocab_size):
    """Normalized JOINT 4-gram [V, V, V, V] over int sequences -- the text side (41^4 = 2.8M cells)."""
    c = np.zeros((vocab_size,) * 4)
    for s in seqs:
        s = s.astype(np.int64)
        if len(s) > 3:
            np.add.at(c, (s[:-3], s[1:-2], s[2:-1], s[3:]), 1.0)
    if c.sum() <= 0:
        sys.exit("empty 4-gram statistics -- wrong data?")
    return c / c.sum()


def fourgram_sparse(seqs, vocab_size):
    """
    Normalized JOINT 4-gram of the audio side, sparse and grouped by prefix for the stagewise
    contraction: returns `(prefix_pair = i*V + j, prefix_k, prefix_id, l, weight)` where the last three
    are per distinct observed 4-gram and `prefix_id` indexes the distinct (i, j, k) prefixes.
    """
    V = vocab_size
    codes = []
    for s in seqs:
        s = s.astype(np.int64)
        if len(s) > 3:
            codes.append(((s[:-3] * V + s[1:-2]) * V + s[2:-1]) * V + s[3:])
    codes = np.concatenate(codes)
    uniq, cnt = np.unique(codes, return_counts=True)
    prefix_code, l = uniq // V, uniq % V
    prefix_uniq, prefix_id = np.unique(prefix_code, return_inverse=True)
    return prefix_uniq // V, prefix_uniq % V, prefix_id, l, cnt / cnt.sum()


def kl(target, pred):
    """KL(target || pred), dropping the constant target-entropy term's sign convention aside."""
    t = target
    return float((t * np.log((t + EPS) / (pred + EPS)))[t > 0].sum())


# ---------------------------------------------------------------- the objective


class SoftMapLoss:
    """
    Differentiable distribution-matching loss for a soft cluster->phoneme map.

    Same loss and `lam` convention as `calc_cheat_seg_identifiability.py`'s "kl" objective, so
    values are comparable across the two scripts.
    """

    def __init__(self, c1, C2, t1, t2, lam, entropy_scale=0.0, C3=None, t3=None, tri_scale=0.0,
                 tri_backoff=0.05, C4=None, t4=None, four_scale=0.0, four_backoff=0.05, chunk=20000,
                 cond_weights=None, collapse=False, lam_z=0.0, z_target=1.0, bigram_ratio=1.0):
        # `cond_weights` = (w1, w2, w3, w4): Task-B mode. The loss is then sum_k w_k C_k with the EXACT
        # conditional KLs C_k = E_p[KL(p(c_k|c_1^{k-1}) || q(c_k|c_1^{k-1}))], each computed from the
        # order-k joint table and ITS OWN marginal over the last symbol (so no marginal-consistency
        # assumption between orders is needed -- the back-off mixing breaks it). C_1 = KL(p1||q1).
        # With consistent marginals this equals sum_n s_n KL(p_n||q_n) for s_n = w_n - w_{n+1}, i.e. the
        # historical loss (lam, 1, tri_scale, four_scale) is w = (lam+1+t+f, 1+t+f, t+f, f).
        # Unlike the s-form it allows w to INCREASE with the order (s_n < 0), which is what a
        # high-order-emphasized sweep needs. None = the historical s-form loss.
        self.cond_weights = None if cond_weights is None else [float(w) for w in cond_weights]
        self.c1 = torch.as_tensor(c1, dtype=torch.float64)
        self.C2 = torch.as_tensor(C2, dtype=torch.float64)
        self.t1 = torch.as_tensor(t1, dtype=torch.float64)
        self.t2 = torch.as_tensor(t2, dtype=torch.float64)
        # --- Task C: collapse-repeats pushforward B appended to the induced frame-level statistics.
        # The model is unchanged; `terms` just maps (q1y, q2y) -> (q1B, q2B) before the KLs, see
        # `_collapse`. Since q2B[c,c] == 0 by construction, any geminate mass on the diagonal of the
        # TEXT bigram would make KL(t2||q2B) infinite for every map, so the diagonal is backed off
        # here (dropped + renormalized; `diag_mass` records what that removed). n <= 2 only.
        self.collapse = bool(collapse)
        # `c1` is normalized over tokens and `C2` over adjacent PAIRS, i.e. over different denominators
        # (they differ by one per utterance). `bigram_ratio` = #pairs / #tokens puts the diagonal of q2
        # back on the token scale so that Z and q1B are exact rather than off by ~1/mean-utt-length.
        self.bigram_ratio = float(bigram_ratio)
        self.lam_z = float(lam_z)
        self.z_target = float(z_target)
        self.diag_mass = 0.0
        if self.collapse:
            assert not tri_scale and not four_scale, "Task C is n <= 2: run with --w w1,w2,0,0"
            d = torch.diagonal(self.t2).clone()
            self.diag_mass = float(d.sum())
            self.t2 = self.t2 - torch.diag(d)
            self.t2 = self.t2 / self.t2.sum()
        self.z_pen = torch.zeros((), dtype=torch.float64)  # set by `terms`, read by `total_of`
        self.z_val = 1.0
        self.lam = lam
        self.entropy_scale = entropy_scale
        # constants of the KLs, so the reported number is a true KL (>= 0, 0 iff matched)
        self.const1 = float((self.t1[self.t1 > 0] * self.t1[self.t1 > 0].log()).sum())
        self.const2 = float((self.t2[self.t2 > 0] * self.t2[self.t2 > 0].log()).sum())
        self.t2m = self.t2.sum(1)  # marginal over the last symbol, for the conditional KL
        self.const2m = float((self.t2m[self.t2m > 0] * self.t2m[self.t2m > 0].log()).sum())
        # optional trigram term, KL(t3 || q3): `C3` = (pair_index, k, weight) over the observed
        # audio triples (see `trigram_sparse`), `t3` the dense text trigram [m, m, m].
        self.tri_scale = tri_scale
        self.tri_backoff = tri_backoff
        if tri_scale:
            pair, k, w = C3
            n = self.c1.shape[0]
            idx = torch.stack([torch.as_tensor(pair, dtype=torch.int64), torch.as_tensor(k, dtype=torch.int64)])
            self.C3 = torch.sparse_coo_tensor(idx, torch.as_tensor(w, dtype=torch.float64), (n * n, n)).coalesce()
            self.t3 = torch.as_tensor(t3, dtype=torch.float64)
            self.const3 = float((self.t3[self.t3 > 0] * self.t3[self.t3 > 0].log()).sum())
            self.t3m = self.t3.sum(2)
            self.const3m = float((self.t3m[self.t3m > 0] * self.t3m[self.t3m > 0].log()).sum())
        # optional 4-gram term, KL(t4 || q4): `C4` from `fourgram_sparse`, `t4` dense [m, m, m, m].
        self.four_scale = four_scale
        self.four_backoff = four_backoff
        self.chunk = chunk
        if four_scale:
            assert tri_scale, "the 4-gram back-off needs the induced trigram, so --tri-scale must be > 0"
            pair, k, pid, l, w = C4
            n = self.c1.shape[0]
            idx = torch.stack([torch.as_tensor(pid, dtype=torch.int64), torch.as_tensor(l, dtype=torch.int64)])
            self.four_dtype = torch.float32
            self.C4f = torch.sparse_coo_tensor(idx, torch.as_tensor(w, dtype=self.four_dtype), (len(pair), n)).coalesce()
            self.four_pair = torch.as_tensor(pair, dtype=torch.int64)
            self.four_k = torch.as_tensor(k, dtype=torch.int64)
            self.t4 = torch.as_tensor(t4, dtype=torch.float64)
            self.const4 = float((self.t4[self.t4 > 0] * self.t4[self.t4 > 0].log()).sum())
            self.t4m = self.t4.sum(3)
            self.const4m = float((self.t4m[self.t4m > 0] * self.t4m[self.t4m > 0].log()).sum())

    def induced(self, M):
        """(q1, q2): the phoneme unigram and joint bigram the assignment `M` induces."""
        return M.t() @ self.c1, M.t() @ self.C2 @ M

    def _collapse(self, q1, q2):
        """
        (q1B, q2B, Z): the collapse-repeats pushforward of the frame-level induced statistics.

        A collapsed symbol is a run start and a collapsed bigram is a run boundary, so with
        `Z = 1 - sum_c q2[c,c]` the expected number of run starts per frame,

            q1B[c]    = (q1[c] - q2[c,c]) / Z
            q2B[c,c'] = q2[c,c'] * (c != c') / Z

        Both are exactly normalized. `bigram_ratio` rescales the diagonal onto the token denominator
        (see __init__); with it, Z and q1B reproduce a direct count of the collapsed sequences to
        ~1e-10 rather than ~1e-3.
        """
        d = torch.diagonal(q2)
        r = self.bigram_ratio
        Z = 1.0 - r * d.sum()
        q1B = (q1 - r * d).clamp_min(0.0)
        q1B = q1B / (q1B.sum() + EPS)
        q2B = (q2 - torch.diag_embed(d)) / (1.0 - d.sum() + EPS)
        return q1B, q2B, Z

    def induced_trigram(self, M, q1, q2):
        """
        q3[a,b,c] = sum_ijk C3[i,j,k] M[i,a] M[j,b] M[k,c], contracted one axis at a time: a sparse
        matmul over the observed triples for `k`, then two small dense einsums. With `tri_backoff` > 0
        the result is mixed with the lower-order induced statistics `q2(a,b) q1(c)` so that a HARD map
        never yields q3 = 0 where t3 > 0 (a single rare text trigram would otherwise make KL(t3||q3)
        infinite for every hard map, the oracle included -- there are 3x fewer audio frames than text
        tokens, so the audio side cannot realize every text trigram).
        """
        n, m = M.shape
        T1 = torch.sparse.mm(self.C3, M).view(n, n, m)  # sum_k C3[i,j,k] M[k,c]  -> [i, j, c]
        T2 = torch.einsum("ijc,jb->ibc", T1, M)  # -> [i, b, c]
        q3 = torch.einsum("ibc,ia->abc", T2, M)  # -> [a, b, c]
        if self.tri_backoff:
            q3 = (1.0 - self.tri_backoff) * q3 + self.tri_backoff * q2[:, :, None] * q1[None, None, :]
        return q3

    def induced_fourgram(self, M, q1, q2, q3):
        """
        q4[a,b,c,d] = sum_ijkl C4[i,j,k,l] M[i,a] M[j,b] M[k,c] M[l,d], one cluster index at a time:
        sparse over the observed prefixes for `l`, a chunked scatter of outer products for `k` (into the
        dense two-cluster-index intermediate [n*n, m, m] -- the unavoidable 3.4 GB / n^2 m^3-FLOP stage),
        then two einsums. Back-off (`four_backoff`) mixes in the conditional q3(a,b,c) q3(b,c,d)/q2(b,c),
        i.e. a proper 4-gram -> trigram back-off, because far more of the text 4-gram mass is unreachable
        from ~560k audio tokens than at trigram order.
        """
        n, m = M.shape
        # the heavy stages run in float32 (half the memory / ~2x the BLAS rate); the KL's log is taken
        # in float64 again. q4 entries are ~1e-6..1e-2, far above float32 resolution.
        Mf = M.to(self.four_dtype)
        T1 = torch.sparse.mm(self.C4f, Mf)  # [P, d]: sum_l C4[(i,j,k), l] M[l, d]
        T2 = torch.zeros(n * n, m, m, dtype=self.four_dtype)
        for s in range(0, T1.shape[0], self.chunk):  # in place: no copy of the 1.7 GB tensor per chunk
            sl = slice(s, s + self.chunk)
            T2.index_add_(0, self.four_pair[sl], Mf[self.four_k[sl]][:, :, None] * T1[sl][:, None, :])
        T3 = torch.matmul(Mf.t().unsqueeze(0), T2.view(n, n, m * m))  # [i, b, (c,d)] = M^T @ T2[i]
        q4 = torch.matmul(Mf.t(), T3.view(n, m * m * m)).view(m, m, m, m).to(torch.float64)
        if self.four_backoff:
            cond = q3[None, :, :, :] / (q2[None, :, :, None] + EPS)  # q(d | b, c)
            q4 = (1.0 - self.four_backoff) * q4 + self.four_backoff * q3[:, :, :, None] * cond
        return q4

    def terms(self, M):
        """
        The forward KLs (KL(t4||q4), KL(t3||q3), KL(t2||q2), KL(t1||q1)); 0 for terms that are off.
        In `cond_weights` mode instead the exact conditional KLs (C4, C3, C2, C1), where
        C_k = KL(t_k||q_k) - KL(marg(t_k)||marg(q_k)) with both marginals over the last symbol.
        """
        q1, q2 = self.induced(M)
        if self.collapse:
            q1, q2, z = self._collapse(q1, q2)
            self.z_val = float(z.detach()) if torch.is_tensor(z) else float(z)
            self.z_pen = self.lam_z * (z - self.z_target) ** 2 if self.lam_z else torch.zeros((), dtype=torch.float64)
        kl1 = -(self.t1 * torch.log(q1 + EPS)).sum() + self.const1
        kl2 = -(self.t2 * torch.log(q2 + EPS)).sum() + self.const2
        zero = torch.zeros((), dtype=torch.float64)
        kl3 = kl4 = zero
        cond = self.cond_weights is not None
        if cond:
            kl2 = kl2 - (-(self.t2m * torch.log(q2.sum(1) + EPS)).sum() + self.const2m)
        if self.tri_scale:
            q3 = self.induced_trigram(M, q1, q2)
            kl3 = -(self.t3 * torch.log(q3 + EPS)).sum() + self.const3
            if cond:
                kl3 = kl3 - (-(self.t3m * torch.log(q3.sum(2) + EPS)).sum() + self.const3m)
            if self.four_scale:
                q4 = self.induced_fourgram(M, q1, q2, q3)
                kl4 = -(self.t4 * torch.log(q4 + EPS)).sum() + self.const4
                if cond:
                    kl4 = kl4 - (-(self.t4m * torch.log(q4.sum(3) + EPS)).sum() + self.const4m)
        return kl4, kl3, kl2, kl1

    def total_of(self, kl4, kl3, kl2, kl1):
        # NB in collapse mode this adds `self.z_pen`, which `terms` set on the SAME `M` -- every call
        # site computes `terms(M)` immediately before `total_of(...)`.
        if self.cond_weights is not None:
            w1, w2, w3, w4 = self.cond_weights
            return w4 * kl4 + w3 * kl3 + w2 * kl2 + w1 * kl1 + self.z_pen
        return self.four_scale * kl4 + self.tri_scale * kl3 + kl2 + self.lam * kl1 + self.z_pen

    def fmt(self, kl4, kl3, kl2, kl1):
        """'4gram x, trigram y, bigram z, unigram w' with only the active terms (cond4.. in cond mode)."""
        names = ("cond4", "cond3", "cond2", "cond1") if self.cond_weights is not None else (
            "4gram", "trigram", "bigram", "unigram")
        parts = []
        if self.four_scale:
            parts.append("%s %.4f" % (names[0], kl4))
        if self.tri_scale:
            parts.append("%s %.4f" % (names[1], kl3))
        parts += ["%s %.4f" % (names[2], kl2), "%s %.4f" % (names[3], kl1)]
        if self.collapse:
            parts.append("Z %.4f" % self.z_val)
            if self.lam_z:
                parts.append("zpen %.4f" % float(self.z_pen))
        return ", ".join(parts)

    def __call__(self, M):
        kls = self.terms(M)
        total = self.total_of(*kls)
        if self.entropy_scale:
            # mean row entropy, pushed down to drive M toward a hard assignment
            ent = -(M * torch.log(M + EPS)).sum(dim=1).mean()
            total = total + self.entropy_scale * ent
        return total

    def hard(self, assign, num_phon):
        """Loss of the hardened map (a plain numpy evaluation, no autograd)."""
        M = np.zeros((len(assign), num_phon))
        M[np.arange(len(assign)), assign] = 1.0
        M = torch.as_tensor(M, dtype=torch.float64)
        kls = self.terms(M)
        return (float(self.total_of(*kls)),) + tuple(float(k) for k in kls)


# ---------------------------------------------------------------- initializations


def init_logits(kind, n, m, rng, args, omap=None, emb=None):
    if kind == "assign":
        a = np.load(args.init_assign)["assign"]
        assert a.shape == (n,), (a.shape, n)
        th = np.full((n, m), -args.init_scale)
        th[np.arange(n), a] = args.init_scale
        return torch.as_tensor(th, dtype=torch.float64)
    if kind == "uniform":
        return torch.as_tensor(rng.normal(0.0, 1e-3, (n, m)), dtype=torch.float64)
    if kind == "random":
        return torch.as_tensor(rng.normal(0.0, args.init_scale, (n, m)), dtype=torch.float64)
    if kind == "kmeans":
        from sklearn.cluster import KMeans

        assert emb is not None, "--init kmeans needs --audio-emb-npz"
        e = emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)
        groups = KMeans(n_clusters=m, n_init=10, random_state=args.seed).fit(e).labels_
        # group g -> phoneme g: an arbitrary correspondence, so this carries grouping structure
        # from the geometry but no label information.
        L = rng.normal(0.0, 1e-3, (n, m))
        L[np.arange(n), groups] += args.init_scale
        return torch.as_tensor(L, dtype=torch.float64)
    if kind in ("oracle", "oracle-corrupt"):
        assert omap is not None
        a = omap.copy()
        if kind == "oracle-corrupt":
            sel = rng.random(n) < args.corrupt
            a[sel] = rng.integers(0, m, int(sel.sum()))
        L = rng.normal(0.0, 1e-3, (n, m))
        L[np.arange(n), a] += args.init_scale
        return torch.as_tensor(L, dtype=torch.float64)
    raise ValueError(kind)


# ---------------------------------------------------------------- optimization


def optimize(loss_fn, logits, args, acc, label, log=True, anchor=None):
    """
    Adam on the logits with a geometric temperature anneal. Returns (assign, report dict).

    `anchor` is an optional `(row_indices, phoneme_indices)` pair held fixed after every step --
    a handful of known cluster->phoneme pairs, the "25 word dictionary" setting.
    """
    logits = logits.clone().requires_grad_(True)
    if anchor is not None:
        rows, cols = anchor
        fixed = torch.full((len(rows), logits.shape[1]), -args.init_scale, dtype=torch.float64)
        fixed[torch.arange(len(rows)), torch.as_tensor(cols)] = args.init_scale
        with torch.no_grad():
            logits[torch.as_tensor(rows)] = fixed
    opt = torch.optim.Adam([logits], lr=args.lr)
    n, m = logits.shape
    taus = np.geomspace(args.tau_start, args.tau_end, args.steps)
    t0 = time.time()
    best = None
    for step in range(args.steps):
        tau = float(taus[step])
        opt.zero_grad()
        M = torch.softmax(logits / tau, dim=1)
        total = loss_fn(M)
        total.backward()
        opt.step()
        if anchor is not None:
            with torch.no_grad():
                logits[torch.as_tensor(anchor[0])] = fixed
        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                M = torch.softmax(logits / tau, dim=1)
                soft = float(loss_fn.total_of(*loss_fn.terms(M)))
                assign = M.argmax(dim=1).numpy()
                hard = loss_fn.hard(assign, m)[0]
                a = acc(assign)
                used = len(set(assign.tolist()))
                mean_max = float(M.max(dim=1).values.mean())
            if best is None or hard < best["hard"]:
                best = dict(step=step, soft=soft, hard=hard, acc=a, assign=assign, used=used)
            if log:
                print(
                    "    %6d  tau %.3f  soft %.4f  hard %.4f  acc %.4f  used %2d/%d  max_p %.2f"
                    % (step, tau, soft, hard, a, used, m, mean_max)
                )
    with torch.no_grad():
        M = torch.softmax(logits / args.tau_end, dim=1)
        assign = M.argmax(dim=1).numpy()
        soft = float(loss_fn.total_of(*loss_fn.terms(M)))
        hard, *hard_terms = loss_fn.hard(assign, m)
    if getattr(args, "keep_best", False) and best is not None and best["hard"] < hard:
        print("  [%s] keeping step %d (hardened %.4f) over the final step (%.4f)"
              % (label, best["step"], best["hard"], hard))
        assign = best["assign"]
        hard, *hard_terms = loss_fn.hard(assign, m)
        soft = best["soft"]
    a = acc(assign)
    print(
        "  [%s] final: soft %.4f | hard %.4f (%s) | acc %.4f -> PER %.1f%% "
        "| phonemes used %d/%d | gap %+.4f | %.0fs"
        % (label, soft, hard, loss_fn.fmt(*hard_terms), a, 100 * (1 - a), len(set(assign.tolist())), m,
           soft - hard, time.time() - t0)
    )
    if soft < hard - 0.02:
        print(
            "    NB the soft loss is %.4f below the hardened one -- the run is partly exploiting the\n"
            "       relaxation (a mixture no discrete map realizes). Anneal further (--tau-end) or\n"
            "       add --entropy-scale before believing the soft number." % (hard - soft)
        )
    return assign, dict(soft=soft, hard=hard, acc=a, assign=assign)


# ---------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-clusters", type=int, default=512)
    p.add_argument("--num-cluster-shards", type=int, default=3)
    p.add_argument("--num-audio-utts", type=int, default=6000, help="matches the sibling scripts")
    p.add_argument("--num-lm-utts", type=int, default=60000)
    p.add_argument(
        "--init",
        default="all",
        choices=["all", "random", "uniform", "kmeans", "oracle", "oracle-corrupt", "assign"],
        help="'all' = uniform + random restarts + the oracle diagnostics (+ kmeans if given an npz)",
    )
    p.add_argument("--audio-emb-npz", help="per-symbol encoder embeddings, for --init kmeans")
    p.add_argument("--init-assign",
                   help="--init assign: npz with `assign`+`keep_p` (e.g. a Task-D HMM solution), "
                        "to refine an existing hard map under this criterion")
    p.add_argument("--restarts", type=int, default=5)
    p.add_argument("--corrupt", type=float, default=0.5, help="--init oracle-corrupt: row fraction randomized")
    p.add_argument(
        "--seed-clusters",
        type=int,
        default=0,
        help="anchor the N most frequent clusters to their true phoneme and hold them there "
             "(measures how weak an initializer has to be before the search takes off)",
    )
    p.add_argument("--seed-sweep", action="store_true", help="sweep --seed-clusters over 0,10,25,50,100,200")
    p.add_argument(
        "--basin",
        action="store_true",
        help="sweep --corrupt over 0.2..1.0 instead: how corrupted a start can the search still climb out of?",
    )
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--tau-start", type=float, default=1.0)
    p.add_argument("--tau-end", type=float, default=0.02, help="0.02 roughly halves the relaxation gap vs 0.1")
    p.add_argument("--init-scale", type=float, default=2.0)
    p.add_argument("--lam", type=float, default=5.0, help="weight of the unigram KL (as in the sibling script)")
    p.add_argument("--entropy-scale", type=float, default=0.0)
    p.add_argument(
        "--tri-scale", type=float, default=0.0,
        help="weight of the trigram term KL(t3||q3) (0 = off, the historical bigram+unigram loss). "
             "By the chain rule the joint trigram KL already contains the bigram and unigram ones, "
             "so the total is (1+1+lam) marginal + (1+1) transition + 1 second-order transition.",
    )
    p.add_argument(
        "--tri-backoff", type=float, default=0.05,
        help="mix q3 with q2(a,b) q1(c) at this weight so a hard map never has q3=0 where t3>0",
    )
    p.add_argument(
        "--four-scale", type=float, default=0.0,
        help="weight of the 4-gram term KL(t4||q4) (0 = off; needs --tri-scale > 0). Seconds per step.",
    )
    p.add_argument(
        "--four-backoff", type=float, default=0.05,
        help="mix q4 with the conditional back-off q3(a,b,c) q3(b,c,d)/q2(b,c) at this weight",
    )
    p.add_argument(
        "--w", default=None,
        help="Task B: effective CONDITIONAL-order weights 'w1,w2,w3,w4' (loss = sum_k w_k C_k with exact "
             "conditional KLs, see SoftMapLoss). Overrides --lam/--tri-scale/--four-scale (a term is computed "
             "iff its weight or a higher one is > 0). The historical (lam 5, bigram 1, tri 10, four 10) "
             "corresponds to '26,21,20,10'; the flat setting is '10,10,10,10'.",
    )
    p.add_argument(
        "--collapse",
        action="store_true",
        help="Task C: append the parameter-free collapse-repeats map B to the induced statistics "
             "(q1,q2 -> q1B,q2B). n <= 2 only. On cheat-seg data this is a measurement of what the "
             "collapse alone costs, since there the audio is already one token per phoneme.",
    )
    p.add_argument("--lam-z", type=float, default=0.0, help="--collapse: weight of (Z - z_target)^2")
    p.add_argument(
        "--z-target",
        type=float,
        default=None,
        help="--collapse: target for Z = 1 - sum_c q2y[c,c] (the collapsed/frame length ratio). "
             "Default = the label-free estimate min(1, E[S]/E[T]) from the two unpaired corpora.",
    )
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--keep-best", action="store_true",
                   help="return the logged step with the LOWEST hardened loss instead of the\n"
                        "final one -- label-free selection, and the anneal tail can drift past\n"
                        "the optimum (RESULTS.md II.D.3). Off by default: existing runs unchanged.")
    p.add_argument("--save-npz", default=None,
                   help="write the loss-selected hardened map (for --init-assign elsewhere)")
    p.add_argument("--cache", default="/var/tmp/cheat_seg_identifiability_cache.pkl")
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    cond_weights = None
    if args.w:
        cond_weights = [float(x) for x in args.w.split(",")]
        assert len(cond_weights) == 4 and min(cond_weights) >= 0, args.w
        # the flags only decide WHICH induced statistics get computed; the weights come from cond_weights
        args.four_scale = 1.0 if cond_weights[3] > 0 else 0.0
        args.tri_scale = 1.0 if (cond_weights[2] > 0 or cond_weights[3] > 0) else 0.0
        print("=== Task-B conditional weighting: w = %s (s-form equivalent: %s) ==="
              % (cond_weights, [cond_weights[i] - (cond_weights[i + 1] if i < 3 else 0.0) for i in range(4)]))
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()
    clus, phon = load_data(args.num_clusters, args.num_cluster_shards, args.cache)
    common = sorted(set(clus) & set(phon))
    eq = [t for t in common if len(clus[t]) == len(phon[t])]
    if not eq:
        sys.exit("no equal-length utterances -- wrong HDFs?")

    # oracle table + accuracy, identical to the sibling scripts
    J = oracle_table(clus, phon, eq, args.num_clusters)
    cf = J.sum(1) / J.sum()
    Pph = J / np.maximum(J.sum(1, keepdims=True), 1)
    oracle_acc = float((cf * Pph.max(1)).sum())
    chance = float((Pph * cf[:, None]).sum(0).max())

    # unpaired split: text statistics come from utterances the audio side never sees
    audio_tags = eq[: args.num_audio_utts]
    aset = set(audio_tags)
    lm_tags = [t for t in common if t not in aset][: args.num_lm_utts]

    c1, C2 = unigram_bigram([clus[t] for t in audio_tags], args.num_clusters)
    t1_full, t2_full = unigram_bigram([phon[t] for t in lm_tags], NUM_PHON)

    # restrict the phoneme axis to the symbols that actually occur: a softmax column for an absent
    # phoneme is mass KL(t||q) cannot see directly (it only sums where t > 0).
    keep_p = np.where(t1_full > 0)[0]
    t1 = t1_full[keep_p]
    t1 = t1 / t1.sum()
    t2 = t2_full[np.ix_(keep_p, keep_p)]
    t2 = t2 / t2.sum()
    n, m = args.num_clusters, len(keep_p)

    p_pos = -np.ones(NUM_PHON, dtype=np.int64)
    p_pos[keep_p] = np.arange(m)
    omap = p_pos[J.argmax(1)]
    if (omap < 0).any():  # a cluster whose majority phoneme never occurs in the text split
        omap = np.where(omap < 0, int(t1.argmax()), omap)
    Pk = Pph[:, keep_p]

    def acc(assign):
        return float((cf * Pk[np.arange(n), assign]).sum())

    print("=== setup ===")
    print("  audio utts %d, disjoint text utts %d" % (len(audio_tags), len(lm_tags)))
    print("  soft map: %d clusters x %d phonemes = %d parameters" % (n, m, n * m))
    print("  oracle acc %.4f -> PER floor %.1f%% | unigram-argmax chance %.4f" % (oracle_acc, 100 * (1 - oracle_acc), chance))

    C3 = t3 = None
    if args.tri_scale:
        C3 = trigram_sparse([clus[t] for t in audio_tags], n)
        t3_full = trigram_dense([phon[t] for t in lm_tags], NUM_PHON)
        t3 = t3_full[np.ix_(keep_p, keep_p, keep_p)]
        t3 = t3 / t3.sum()
        n_text_tok = sum(max(len(phon[t]) - 2, 0) for t in lm_tags)
        n_audio_tok = sum(max(len(clus[t]) - 2, 0) for t in audio_tags)
        print(
            "  trigram term on (scale %.2f, backoff %.3f): text %d trigram tokens -> %d/%d nonzero cells "
            "(%.1f%%); audio %d trigram tokens -> %d distinct cluster triples"
            % (args.tri_scale, args.tri_backoff, n_text_tok, int((t3 > 0).sum()), m ** 3,
               100 * (t3 > 0).mean(), n_audio_tok, len(C3[2]))
        )

    C4 = t4 = None
    if args.four_scale:
        C4 = fourgram_sparse([clus[t] for t in audio_tags], n)
        t4_full = fourgram_dense([phon[t] for t in lm_tags], NUM_PHON)
        t4 = t4_full[np.ix_(keep_p, keep_p, keep_p, keep_p)]
        t4 = t4 / t4.sum()
        print(
            "  4-gram term on (scale %.2f, backoff %.3f): text %d/%d nonzero cells (%.1f%%); audio %d "
            "distinct cluster 4-grams over %d distinct prefixes"
            % (args.four_scale, args.four_backoff, int((t4 > 0).sum()), m ** 4, 100 * (t4 > 0).mean(),
               len(C4[4]), len(C4[0]))
        )

    z_target, bigram_ratio = 1.0, 1.0
    if args.collapse:
        # label-free: mean text length over mean audio length, from the two UNPAIRED corpora, clamped
        # to 1 (the collapse can only shorten). On cheat-seg both the unpaired (0.78) and the true
        # paired (0.98) ratio are < 1, so the target is 1.0 either way; on real clus128 it is ~0.38.
        mean_s = float(np.mean([len(phon[t]) for t in lm_tags]))
        mean_t = float(np.mean([len(clus[t]) for t in audio_tags]))
        n_tok = sum(len(clus[t]) for t in audio_tags)
        bigram_ratio = sum(max(len(clus[t]) - 1, 0) for t in audio_tags) / n_tok
        z_target = min(1.0, mean_s / mean_t) if args.z_target is None else args.z_target
        print("  collapse (Task C): E[S] %.2f / E[T] %.2f = %.4f -> z_target %.4f, lam_z %.3g"
              % (mean_s, mean_t, mean_s / mean_t, z_target, args.lam_z))

    loss_fn = SoftMapLoss(c1, C2, t1, t2, lam=args.lam, entropy_scale=args.entropy_scale,
                          C3=C3, t3=t3, tri_scale=args.tri_scale, tri_backoff=args.tri_backoff,
                          C4=C4, t4=t4, four_scale=args.four_scale, four_backoff=args.four_backoff,
                          cond_weights=cond_weights, collapse=args.collapse, lam_z=args.lam_z,
                          z_target=z_target, bigram_ratio=bigram_ratio if args.collapse else 1.0)
    if args.collapse:
        print("  text bigram diagonal (geminates) backed off: %.5f of the mass dropped + renormalized"
              % loss_fn.diag_mass)
    o_hard, *o_terms = loss_fn.hard(omap, m)
    print(
        "  reference losses (same convention as calc_cheat_seg_identifiability.py):\n"
        "    oracle map          loss %.4f (%s)  acc %.4f\n"
        "    real text vs itself loss %.4f (a perfect match would be 0)"
        % (o_hard, loss_fn.fmt(*o_terms), acc(omap), 0.0)
    )
    print("    for comparison, that script's hard hill-climb: best random restart 0.206 @ acc 0.045,")
    print("    from the oracle 0.073 @ acc 0.657 -- it never *increased* accuracy from a random start.")

    emb = None
    if args.audio_emb_npz:
        emb = np.load(args.audio_emb_npz)["emb"]
        assert emb.shape[0] == n, "npz has %d rows, expected %d" % (emb.shape[0], n)

    order = np.argsort(-cf)  # clusters by frame mass, for the anchor sets

    def anchor_for(k):
        """(anchor, frame mass, accuracy the anchors alone already account for)."""
        if not k:
            return None, 0.0, 0.0
        rows = order[:k]
        alone = float((cf[rows] * Pk[rows, omap[rows]]).sum())
        return (rows, omap[rows]), float(cf[rows].sum()), alone

    if args.seed_sweep:
        kinds = [("random", args.corrupt)] * 6
        seed_counts = [0, 10, 25, 50, 100, 200]
    elif args.basin:
        kinds = [("oracle-corrupt", c) for c in (0.2, 0.4, 0.6, 0.8, 0.9, 1.0)]
    elif args.init == "all":
        kinds = [(k, args.corrupt) for k in
                 ["uniform"] + ["random"] * args.restarts
                 + (["kmeans"] if emb is not None else []) + ["oracle", "oracle-corrupt"]]
    else:
        kinds = [(args.init, args.corrupt)] * (args.restarts if args.init == "random" else 1)

    if not args.seed_sweep:
        seed_counts = [args.seed_clusters] * len(kinds)

    results = []
    names = [k for k, _ in kinds]
    for i, (kind, corrupt) in enumerate(kinds):
        args.corrupt = corrupt
        label = kind if names.count(kind) == 1 else "%s-%s" % (
            kind, ("%.1f" % corrupt) if args.basin else sum(k == kind for k in names[:i])
        )
        logits = init_logits(kind, n, m, rng, args, omap=omap, emb=emb)
        anchor, anchor_mass, anchor_alone = anchor_for(seed_counts[i])
        if args.seed_sweep:
            label = "seed-%d" % seed_counts[i]
        with torch.no_grad():
            a0 = acc(torch.softmax(logits / args.tau_start, dim=1).argmax(dim=1).numpy())
        print(
            "\n=== %s (init acc %.4f%s) ==="
            % (label, a0, ", %d anchored = %.1f%% of frames" % (seed_counts[i], 100 * anchor_mass)
               if seed_counts[i] else "")
        )
        _, rep = optimize(loss_fn, logits, args, acc, label, anchor=anchor)
        rep["label"] = label
        rep["anchor_mass"] = anchor_mass
        rep["anchor_alone"] = anchor_alone
        rep["n_anchor"] = seed_counts[i]
        rep["init_acc"] = a0
        results.append(rep)

    if args.save_npz:
        best = min(results, key=lambda r: r["hard"])
        np.savez(args.save_npz, assign=best["assign"], keep_p=keep_p, hard=best["hard"],
                 acc=best["acc"], label=best["label"])
        print("\n  wrote %s (%s, hard %.4f, acc %.4f)"
              % (args.save_npz, best["label"], best["hard"], best["acc"]))

    print("\n=== summary (sorted by the hardened loss, which is what a real decoder would use) ===")
    print("  %-18s %8s %8s %8s %8s" % ("init", "init_acc", "soft", "hard", "acc"))
    for r in sorted(results, key=lambda r: r["hard"]):
        print("  %-18s %8.4f %8.4f %8.4f %8.4f" % (r["label"], r["init_acc"], r["soft"], r["hard"], r["acc"]))
    print("  %-18s %8s %8s %8.4f %8.4f" % ("(oracle map)", "--", "--", o_hard, acc(omap)))

    if args.seed_sweep:
        print("\n=== how weak an anchor is enough? (anchored rows held fixed throughout) ===")
        print(
            "  %-10s %10s %12s %10s %8s %10s"
            % ("anchored", "of frames", "anchors only", "final_acc", "gained", "hard_loss")
        )
        for r in results:
            print(
                "  %-10d %9.1f%% %12.4f %10.4f %+8.4f %10.4f"
                % (r["n_anchor"], 100 * r["anchor_mass"], r["anchor_alone"], r["acc"],
                   r["acc"] - r["anchor_alone"], r["hard"])
            )
        print("  --> 'anchors only' is the accuracy the fixed rows already account for, so 'gained' is")
        print("      what the unpaired statistics contributed on top of the supervision.")

    if args.basin:
        print("\n=== basin: does the search climb UP in accuracy, which the hard hill-climb never did? ===")
        print("  %-10s %9s %9s %8s %9s" % ("corrupt", "init_acc", "final_acc", "delta", "hard_loss"))
        for r in results:
            print(
                "  %-10s %9.4f %9.4f %+8.4f %9.4f"
                % (r["label"].split("-")[-1], r["init_acc"], r["acc"], r["acc"] - r["init_acc"], r["hard"])
            )
        gains = [r["acc"] - r["init_acc"] for r in results]
        if max(gains) > 0.02:
            print("  --> yes: the continuous relaxation moves uphill in accuracy, unlike the discrete climb.")
        print("  --> read off the weakest init that still gains: that is the accuracy an initializer must supply.")

    unsup = [r for r in results if r["label"].startswith(("random", "uniform", "kmeans"))]
    if unsup:
        b = min(unsup, key=lambda r: r["hard"])
        print(
            "\n  best unsupervised init: %s -- hard loss %.4f vs the oracle's %.4f, acc %.4f vs %.4f"
            % (b["label"], b["hard"], o_hard, b["acc"], oracle_acc)
        )
        if b["hard"] < o_hard and b["acc"] < acc(omap):
            print("  --> the loss still prefers a wrong map: an unsupervised optimum beats the truth's loss.")
        elif b["acc"] > chance + 0.05:
            print("  --> an unsupervised run beat chance: the soft relaxation reaches what the hard climb could not.")
        else:
            print("  --> at chance: the loss ranks the truth correctly but the search still cannot get there.")

    print("\n(%.0fs)" % (time.time() - t0))


if __name__ == "__main__":
    main()
