#!/usr/bin/env python3
"""
How much of the phoneme labeling do the output-statistics losses actually pin down?

``models/train_steps/output_stats.py`` adds two corpus-level matching terms to the unsupervised
setup: a phoneme **unigram** KL and a **bigram** (phonotactics) KL between what the text CTC head
emits on audio and the unpaired phoneme text. Both are invariant under some relabelings of the
phoneme inventory, and a relabeling the loss cannot see is a solution the training can converge to.
This script measures, on the real phoneme statistics, how much ambiguity each term leaves.

Notation: ``p`` is the reference phoneme **unigram** and ``B`` the reference **bigram** (the code
calls them ``p`` and ``b``); ``q`` and ``Q`` are the unigram/bigram a *relabeled* model would emit.

For a relabeling ``sigma`` -- the model emits symbol ``sigma(c)`` whenever the truth is phone ``c``
-- the emitted frequency of ``sigma(c)`` is the true frequency of ``c``, so
``q[sigma(c)] = p[c]`` and ``Q[sigma(a), sigma(b)] = B[a, b]`` (that is what ``relabel_uni`` /
``relabel_bi`` compute). The cost the loss would see is then ``KL(p || q)`` resp. ``KL(B || Q)``,
reference first, as in ``output_stats._kl``.

A cost of ~0 means the loss cannot tell that wrong labeling apart from the correct one: it is an
equally good optimum for that term, and training can converge to it. Such a relabeling is called
**blind** below. Since there is no principled cutoff for "~0", the counts are reported over a sweep
of thresholds -- the shape of that sweep is the result, not any single row. For scale, a *random*
relabeling costs a median of 1.74 nats under the unigram term, so a swap below 0.001 is some three
orders of magnitude cheaper than a typical wrong labeling.

Reported:

1. random relabelings -- the coarse discriminative power of each term;
2. single transpositions (all phoneme pairs) -- the *local* ambiguity that matters in practice, since
   gradient descent reaches a neighbouring labeling far more easily than a random one. This is the
   experiment that decides whether ``bigram_scale`` is needed: the unigram term is blind to swapping
   two phonemes of near-equal frequency;
3. the residual -- transpositions that are cheap under *both* terms, i.e. what neither term can fix.

Note this measures the information available in the statistic, i.e. an upper bound on what the loss
can exploit: the loss's *predicted* bigram is a soft frame-level estimate (see
``output_stats._soft_bigram``), not an exact token bigram.

Usage::

    ./sis  # not needed; this is a plain script
    python3 calc_output_stats_identifiability.py [--stm PATH] [--vocab PATH] [--num-perms N]
"""

import argparse
import ast
import itertools
from typing import Dict, List, Tuple

import numpy as np

DEFAULT_STM = (
    "/work/asr4/schmitt/sisyphus_work_dirs/2026_04_09_unsupervised_asr/i6_core/text/convert/"
    "TextDictToStmJob.GlEbcEUNuQ2r/output/corpus.stm"
)
DEFAULT_VOCAB = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_experiments/users/schmitt/datasets/"
    "utils/phonemize/PhonemizeTextDataJob.rnyWlvnkNbNd/output/phoneme_vocab.txt"
)


def read_vocab(path: str) -> Dict[str, int]:
    return ast.literal_eval(open(path).read())


def read_stm(path: str, vocab: Dict[str, int]) -> List[List[int]]:
    """Reference phoneme sequences, one list of ids per utterance."""
    seqs = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith(";;"):
            continue
        # <id> <ch> <spk> <start> <end> <label> tokens...
        parts = line.split()
        if len(parts) < 7:
            continue
        toks = parts[6:]
        if toks and toks[0].startswith("<") and toks[0].endswith(">"):
            toks = toks[1:]
        ids = [vocab[t] for t in toks if t in vocab]
        if ids:
            seqs.append(ids)
    return seqs


def statistics(seqs: List[List[int]], num_phon: int) -> Tuple[np.ndarray, np.ndarray]:
    """Token unigram ``[P]`` and within-utterance token bigram ``[P, P]``, both normalized."""
    uni = np.zeros(num_phon)
    bi = np.zeros((num_phon, num_phon))
    for s in seqs:
        a = np.asarray(s)
        uni += np.bincount(a, minlength=num_phon)
        if a.size > 1:
            np.add.at(bi, (a[:-1], a[1:]), 1.0)
    return uni / uni.sum(), bi / bi.sum()


def kl(target: np.ndarray, pred: np.ndarray) -> float:
    """KL(target || pred), matching output_stats._kl (entries with no target mass are skipped)."""
    m = target > 0
    return float((target[m] * np.log(target[m] / np.clip(pred[m], 1e-12, None))).sum())


def relabel_uni(p: np.ndarray, perm: np.ndarray) -> np.ndarray:
    q = np.empty_like(p)
    q[perm] = p
    return q


def relabel_bi(b: np.ndarray, perm: np.ndarray) -> np.ndarray:
    q = np.empty_like(b)
    q[np.ix_(perm, perm)] = b
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stm", default=DEFAULT_STM)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--num-perms", type=int, default=20000)
    ap.add_argument("--free-threshold", type=float, default=1e-3, help="KL below which a term is blind")
    args = ap.parse_args()

    vocab = read_vocab(args.vocab)
    inv = {v: k for k, v in vocab.items()}
    num_phon = max(vocab.values()) + 1
    seqs = read_stm(args.stm, vocab)
    p, b = statistics(seqs, num_phon)
    num_tokens = sum(len(s) for s in seqs)
    print(f"{len(seqs)} utterances, {num_tokens} phoneme tokens, {num_phon} phoneme ids")
    print(f"unigram: H={-(p[p>0]*np.log2(p[p>0])).sum():.3f} bits, perplexity={2**(-(p[p>0]*np.log2(p[p>0])).sum()):.1f}")
    nz = b > 0
    print(f"bigram : {int(nz.sum())} of {num_phon**2} cells observed, "
          f"H={-(b[nz]*np.log2(b[nz])).sum():.3f} bits")
    # how far the real bigram is from the product of its marginals = the phonotactic information
    print(f"phonotactic information I(prev;next) = "
          f"{-(b[nz]*np.log2(np.outer(b.sum(1), b.sum(0))[nz]/b[nz])).sum():.4f} bits\n")

    rng = np.random.default_rng(0)

    # --- 1. random relabelings -----------------------------------------------------------------
    uni_costs, bi_costs = [], []
    for _ in range(args.num_perms):
        perm = rng.permutation(num_phon)
        uni_costs.append(kl(p, relabel_uni(p, perm)))
        bi_costs.append(kl(b, relabel_bi(b, perm)))
    uni_costs, bi_costs = np.array(uni_costs), np.array(bi_costs)
    print(f"=== random relabelings (n={args.num_perms}) ===")
    for nm, c in (("unigram", uni_costs), ("bigram", bi_costs)):
        print(f"  {nm:>7}: min={c.min():.4f} 1%={np.percentile(c,1):.4f} "
              f"median={np.percentile(c,50):.3f} max={c.max():.3f}")
    print(f"  blind (<{args.free_threshold}): unigram {int((uni_costs<args.free_threshold).sum())}, "
          f"bigram {int((bi_costs<args.free_threshold).sum())}\n")

    # Symbols with no mass in this reference cannot be constrained by *any* statistic -- swapping
    # two of them is free but also vacuous (neither is ever emitted). Flag them so the residual
    # below is not misread as a real ambiguity.
    absent = [i for i in range(num_phon) if p[i] == 0.0]
    if absent:
        print(f"symbols with zero mass in this reference (unconstrainable, but vacuous): "
              f"{', '.join(inv[i] for i in absent)}\n")
    absent_set = set(absent)

    # --- 2. single transpositions: the local ambiguity that training can actually fall into -----
    pairs = list(itertools.combinations(range(num_phon), 2))
    rows = []
    for i, j in pairs:
        perm = np.arange(num_phon)
        perm[[i, j]] = perm[[j, i]]
        rows.append((kl(p, relabel_uni(p, perm)), kl(b, relabel_bi(b, perm)), i, j))
    rows.sort()
    uc = np.array([r[0] for r in rows])
    bc = np.array([r[1] for r in rows])
    print(f"=== single transpositions (all {len(pairs)} phoneme pairs) ===")
    print(f"  unigram blind (<{args.free_threshold}): {int((uc<args.free_threshold).sum())} of {len(pairs)}")
    print(f"  bigram  blind (<{args.free_threshold}): {int((bc<args.free_threshold).sum())} of {len(pairs)}")
    blind_uni = uc < args.free_threshold
    if blind_uni.any():
        rescued = bc[blind_uni]
        print(f"\n  Of the {int(blind_uni.sum())} swaps the UNIGRAM cannot see, the BIGRAM cost is:")
        print(f"    min={rescued.min():.4f}  median={np.median(rescued):.4f}  max={rescued.max():.4f}")
        print(f"    still blind under bigram too: {int((rescued<args.free_threshold).sum())}")
        print(f"\n  worst offenders for the unigram, and what the bigram charges them:")
        print(f"    {'swap':>14} {'unigram KL':>11} {'bigram KL':>10}")
        for u, bb, i, j in rows[:12]:
            print(f"    {inv[i]+' <-> '+inv[j]:>14} {u:>11.6f} {bb:>10.4f}")

    # --- 3. residual: cheap under BOTH ----------------------------------------------------------
    both = [(u, bb, i, j) for u, bb, i, j in rows if u < args.free_threshold and bb < args.free_threshold]
    print(f"\n=== residual ambiguity: transpositions cheap under BOTH terms ===")
    real = [(u, bb, i, j) for u, bb, i, j in both if i not in absent_set or j not in absent_set]
    if not both:
        print("  none -- every single-phoneme swap is visible to at least one term")
    else:
        for u, bb, i, j in both:
            vac = " (vacuous: both symbols never occur)" if i in absent_set and j in absent_set else ""
            print(f"  {inv[i]} <-> {inv[j]}: unigram={u:.6f} bigram={bb:.6f}{vac}")
    print(f"  -> {len(real)} non-vacuous residual ambiguities")

    # margin: how visible are the swaps to the bigram at looser thresholds
    print(f"\n=== bigram margin over all {len(pairs)} transpositions ===")
    for thr in (1e-3, 1e-2, 5e-2, 1e-1):
        print(f"  bigram KL < {thr:<5}: {int((bc<thr).sum()):>4}   |   unigram KL < {thr:<5}: {int((uc<thr).sum()):>4}")


if __name__ == "__main__":
    main()
