#!/usr/bin/env python3
"""
Stage A of the LM-guided pipeline: is the model's output a *relabeled* version of the truth?

The unsupervised setups reach ~108% PER with ~59% substitutions, i.e. the emission rate is roughly
right but the labeling is wrong. Two very different things could be going on:

  (a) the acoustic model is essentially fine and only the label<->phoneme correspondence is
      permuted, in which case a single relabeling fixes everything at once, or
  (b) the model's errors are not a relabeling at all (many-to-one collapse, context confusion),
      in which case relabeling its own output cannot help and better *hypotheses* are needed.

This script decides between them. It searches -- **without using the reference** -- for the
relabeling ``sigma`` of the output inventory that makes ``sigma(hypothesis)`` most likely under a
phoneme n-gram LM estimated from *unpaired training text*, hill-climbing over transpositions. The
reference is then used only to *evaluate* the result (PER before vs. after).

That search landscape is the one ``calc_output_stats_identifiability.py`` mapped: 819 of 820 single
swaps change the bigram KL by >=0.001 nats, so there is real gradient to climb.

Usage::

    python3 calc_relabeling_search.py [--hyp SEARCH_OUT] [--restarts N] [--order 3]
"""

import argparse
import ast
import gzip
import random
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_HYP = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/output/unsup_denoising_audio_cluster_and_phoneme/"
    "librispeech/config_librispeech_960_wo_sil_from_unsupervised_ctc_only_v1/"
    "baseline_low-lr-0.0001_peak-lr-0.0001_freeze-enc_freeze-emb_out-stats-uni/recog/20/dev-other/search_out.py.gz"
)
DEFAULT_STM = (
    "/work/asr4/schmitt/sisyphus_work_dirs/2026_04_09_unsupervised_asr/i6_core/text/convert/"
    "TextDictToStmJob.GlEbcEUNuQ2r/output/corpus.stm"
)
DEFAULT_VOCAB = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_experiments/users/schmitt/datasets/"
    "utils/phonemize/PhonemizeTextDataJob.rnyWlvnkNbNd/output/phoneme_vocab.txt"
)
# train-960 phoneme text (unpaired) -- the LM source. Using the dev reference here would leak.
DEFAULT_LM_HDF_GLOB = (
    "/u/schmitt/experiments/2026_04_09_unsupervised_asr/work/i6_experiments/users/schmitt/datasets/"
    "utils/phonemize/DumpPhonemeIndicesToHdfJob.V3rHWoSUS9Hf/output/data_*.hdf"
)


def read_vocab(path: str) -> Dict[str, int]:
    return ast.literal_eval(open(path).read())


def read_hyps(path: str) -> Dict[str, List[str]]:
    d = ast.literal_eval(gzip.open(path, "rt").read())
    out = {}
    for tag, v in d.items():
        best = v[0][1] if isinstance(v, list) else v
        out[tag] = best.split()
    return out


def read_refs(path: str) -> Dict[str, List[str]]:
    refs = {}
    for line in open(path):
        parts = line.split()
        if len(parts) < 7 or line.startswith(";;"):
            continue
        toks = parts[6:]
        if toks and toks[0].startswith("<") and toks[0].endswith(">"):
            toks = toks[1:]
        refs[parts[0]] = toks
    return refs


def load_lm_counts(glob_pat: str, num_phon: int, order: int) -> np.ndarray:
    """N-gram counts over unpaired *training* phoneme text, as a flat array of shape [V]*order."""
    import glob

    import h5py

    counts = np.zeros((num_phon,) * order, dtype=np.float64)
    num_seqs = 0
    for path in sorted(glob.glob(glob_pat)):
        with h5py.File(path, "r") as f:
            data = np.asarray(f["inputs"]).astype(np.int64)
            lens = np.asarray(f["seqLengths"])[:, 0]
        off = 0
        for L in lens:
            s = data[off : off + L]
            off += L
            num_seqs += 1
            if s.size < order:
                continue
            idx = tuple(s[i : s.size - order + 1 + i] for i in range(order))
            np.add.at(counts, idx, 1.0)
    print(f"LM: {num_seqs} training sequences, {int(counts.sum())} {order}-grams")
    return counts


def make_logprob(counts: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Add-alpha smoothed conditional log-probs of the last symbol given the preceding context."""
    sm = counts + alpha
    return np.log(sm / sm.sum(axis=-1, keepdims=True))


def score(seqs: Sequence[np.ndarray], logp: np.ndarray, order: int, perm: np.ndarray) -> float:
    """Total LM log-prob of the relabeled hypotheses. Higher is better."""
    total = 0.0
    for s in seqs:
        if s.size < order:
            continue
        t = perm[s]
        total += float(logp[tuple(t[i : t.size - order + 1 + i] for i in range(order))].sum())
    return total


def per(hyps: Dict[str, List[str]], refs: Dict[str, List[str]]) -> float:
    """Levenshtein PER over the shared keys (sclite-comparable up to tie-breaking)."""
    err = ref_len = 0
    for tag, h in hyps.items():
        r = refs.get(tag)
        if r is None:
            continue
        prev = list(range(len(h) + 1))
        for i, rt in enumerate(r, 1):
            cur = [i] + [0] * len(h)
            for j, ht in enumerate(h, 1):
                cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rt != ht))
            prev = cur
        err += prev[-1]
        ref_len += len(r)
    return 100.0 * err / max(ref_len, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hyp", default=DEFAULT_HYP)
    ap.add_argument("--stm", default=DEFAULT_STM)
    ap.add_argument("--vocab", default=DEFAULT_VOCAB)
    ap.add_argument("--lm-hdf", default=DEFAULT_LM_HDF_GLOB)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--max-seqs", type=int, default=0, help="0 = all; smaller is faster for the search")
    args = ap.parse_args()

    vocab = read_vocab(args.vocab)
    inv = {v: k for k, v in vocab.items()}
    num_phon = max(vocab.values()) + 1
    hyps = read_hyps(args.hyp)
    refs = read_refs(args.stm)
    print(f"{len(hyps)} hypotheses, {len(refs)} references, {num_phon} phoneme ids")

    logp = make_logprob(load_lm_counts(args.lm_hdf, num_phon, args.order))
    seq_ids = [np.array([vocab[t] for t in h if t in vocab], dtype=np.int64) for h in hyps.values()]
    if args.max_seqs:
        seq_ids = seq_ids[: args.max_seqs]

    base_per = per(hyps, refs)
    identity = np.arange(num_phon)

    def per_token(seqs, perm):
        tot = sum(max(s.size - args.order + 1, 0) for s in seqs)
        return score(seqs, logp, args.order, perm) / max(tot, 1)

    # Control: the LM score of the *reference* transcripts. If a relabeled hypothesis scores better
    # per token than real text does, the search is gaming the LM rather than recovering the truth.
    ref_ids = [np.array([vocab[t] for t in r if t in vocab], dtype=np.int64) for r in refs.values()]
    if args.max_seqs:
        ref_ids = ref_ids[: args.max_seqs]
    ref_lp = per_token(ref_ids, identity)

    print(f"\nbaseline (no relabeling): PER={base_per:.2f}  LM logprob/token={per_token(seq_ids, identity):.4f}")
    print(f"CONTROL -- real reference text:            LM logprob/token={ref_lp:.4f}")

    # --- hill-climb over transpositions, unsupervised (LM score only) --------------------------
    rng = random.Random(0)
    best_perm, best_score = identity.copy(), score(seq_ids, logp, args.order, identity)
    for r in range(args.restarts):
        perm = identity.copy() if r == 0 else np.random.default_rng(r).permutation(num_phon)
        cur = score(seq_ids, logp, args.order, perm)
        improved = True
        while improved:
            improved = False
            pairs = [(i, j) for i in range(num_phon) for j in range(i + 1, num_phon)]
            rng.shuffle(pairs)
            for i, j in pairs:
                cand = perm.copy()
                cand[[i, j]] = cand[[j, i]]
                s = score(seq_ids, logp, args.order, cand)
                if s > cur:
                    perm, cur, improved = cand, s, True
        tag = "identity-start" if r == 0 else f"random-start-{r}"
        print(f"  {tag:>16}: LM logprob={cur:,.0f}")
        if cur > best_score:
            best_perm, best_score = perm, cur

    relabeled = {t: [inv[int(best_perm[vocab[x]])] if x in vocab else x for x in h] for t, h in hyps.items()}
    new_per = per(relabeled, refs)
    moved = [(inv[i], inv[int(best_perm[i])]) for i in range(num_phon) if best_perm[i] != i]
    print(f"\nbest relabeling: LM logprob/token={per_token(seq_ids, best_perm):.4f}"
          f"  ({len(moved)} of {num_phon} symbols moved)")
    print(f"PER  before={base_per:.2f}  after={new_per:.2f}   ({base_per - new_per:+.2f})")
    if per_token(seq_ids, best_perm) > ref_lp:
        print("  !! the relabeled hypotheses are MORE LM-likely than real text -- the search is\n"
              "     gaming the LM (mapping onto frequent, predictable phonemes), not recovering truth.")
    if moved:
        print("  mapping changes: " + ", ".join(f"{a}->{b}" for a, b in moved[:20]) + ("..." if len(moved) > 20 else ""))


if __name__ == "__main__":
    main()
