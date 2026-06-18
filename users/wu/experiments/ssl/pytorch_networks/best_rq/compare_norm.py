"""
Compare per-utterance vs corpus-level (global, fixed) input normalization for the BEST-RQ
quantizer, on real LibriSpeech features. Global norm is sequence-independent (a fixed affine),
so a given acoustic frame always maps to the same code -- conceptually a cleaner SSL target.

Run: PYTHONPATH=recipe:recipe/i6_models <conda-python> .../best_rq/compare_norm.py --num-utts 500
"""

from __future__ import annotations

import argparse
import torch

from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.check_codebook_coverage import (
    _load_logmel, _report,
)
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.input_norm import (
    masked_mean_var_norm,
)
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.quantizer import (
    MultiCodebookRandomProjectionQuantizer,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-utts", type=int, default=500)
    ap.add_argument("--split", type=str, default="validation.clean")
    ap.add_argument("--codebook-dim", type=int, default=16)
    args = ap.parse_args()
    V, N = 8192, 4
    q = MultiCodebookRandomProjectionQuantizer(
        input_dim=80, stack_size=4, codebook_dim=args.codebook_dim, vocab_size=V, num_codebooks=N, seed=42
    )

    # Pass 1: corpus-level per-feature mean/std over all valid frames.
    s = torch.zeros(80, dtype=torch.float64)
    ss = torch.zeros(80, dtype=torch.float64)
    cnt = 0
    cache = []
    for feats, lens in _load_logmel(args.num_utts, args.split):
        cache.append((feats, lens))
        for i, n in enumerate(lens.tolist()):
            v = feats[i, :n].double()
            s += v.sum(0); ss += (v * v).sum(0); cnt += n
    g_mean = (s / cnt).float()
    g_std = torch.sqrt((ss / cnt).float() - g_mean ** 2 + 1e-5)
    print(f"[global stats] frames={cnt} mean[:3]={g_mean[:3].tolist()} std[:3]={g_std[:3].tolist()}")

    counts_utt = [torch.zeros(V, dtype=torch.int64) for _ in range(N)]
    counts_glob = [torch.zeros(V, dtype=torch.int64) for _ in range(N)]
    for feats, lens in cache:
        t2 = feats.shape[1] // 4
        valid = torch.arange(t2)[None, :] < (lens // 4)[:, None]
        # per-utterance
        tg_u, _ = q(masked_mean_var_norm(feats, lens), lens)
        # global fixed (zero out padding to match)
        normed_g = (feats - g_mean) / g_std
        m = (torch.arange(feats.shape[1])[None, :] < lens[:, None]).unsqueeze(-1)
        normed_g = normed_g * m
        tg_g, _ = q(normed_g, lens)
        for n in range(N):
            counts_utt[n] += torch.bincount(tg_u[..., n][valid], minlength=V)
            counts_glob[n] += torch.bincount(tg_g[..., n][valid], minlength=V)

    _report("PER-UTTERANCE norm", counts_utt, V)
    _report("GLOBAL (corpus-level fixed) norm", counts_glob, V)


if __name__ == "__main__":
    main()
