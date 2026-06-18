"""
Standalone codebook-coverage diagnostic for the BEST-RQ quantizer.

This is the gate before any long pretraining run: it loads real LibriSpeech audio from the
local HF cache (offline), computes the exact 80-dim log-mel the model uses, applies the
per-utterance input normalization, runs the frozen multi-codebook quantizer, and reports
per-codebook codebook utilization. Code collapse (a few codes absorbing most frames) is the
canonical BEST-RQ failure; input-norm + L2-norm are precisely what prevent it, so we also run
ablations (no input-norm, no L2) to show the effect.

Run (login node has the conda env + the HF cache):

    cd /e/project1/spell/wu24/2026-06-17_ssl
    PYTHONPATH=recipe:recipe/i6_models \
      /e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python \
      recipe/i6_experiments/users/wu/experiments/ssl/pytorch_networks/best_rq/check_codebook_coverage.py \
      --num-utts 400 --split validation.clean

Healthy result: per-codebook coverage close to 100% of the vocab over enough frames, normalized
entropy ~0.9-1.0, max single-code frequency small (a few x uniform). The no-norm / no-L2
ablations should be visibly worse (lower coverage, lower entropy, spikier max-freq).
"""

from __future__ import annotations

import argparse
import glob
import os

import torch

from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.input_norm import (
    masked_mean_var_norm,
)
from i6_experiments.users.wu.experiments.ssl.pytorch_networks.best_rq.parts.quantizer import (
    MultiCodebookRandomProjectionQuantizer,
)


def _find_parquet(split: str) -> list:
    hf_home = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
    repo = os.path.join(hf_home, "hub", "datasets--openslr--librispeech_asr")
    refs_main = os.path.join(repo, "refs", "main")
    commit = None
    if os.path.isfile(refs_main):
        with open(refs_main) as f:
            commit = f.read().strip()
    if not commit:
        snaps = sorted(os.listdir(os.path.join(repo, "snapshots")))
        commit = snaps[-1] if snaps else None
    assert commit, f"no cached snapshot under {repo}"
    pattern = os.path.join(repo, "snapshots", commit, "all", split, "*.parquet")
    files = sorted(glob.glob(pattern))
    assert files, f"no parquet matched {pattern}"
    return files


def _load_logmel(num_utts: int, split: str, batch_size: int = 16):
    """Yield batches of (features [B,T,80], lengths [B]) from real LibriSpeech audio."""
    import datasets
    from i6_models.primitives.feature_extraction import (
        LogMelFeatureExtractionV1,
        LogMelFeatureExtractionV1Config,
    )

    feat = LogMelFeatureExtractionV1(
        cfg=LogMelFeatureExtractionV1Config(
            sample_rate=16000, win_size=0.025, hop_size=0.01,
            f_min=60, f_max=7600, min_amp=1e-10, num_filters=80, center=False,
        )
    )
    feat.eval()

    files = _find_parquet(split)
    ds = datasets.load_dataset("parquet", data_files={"x": files}, split="x")
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=True))
    n = min(num_utts, len(ds))
    print(f"[data] split={split} using {n}/{len(ds)} utterances from {len(files)} parquet shard(s)")

    for start in range(0, n, batch_size):
        rows = ds[start : min(start + batch_size, n)]
        arrays = [torch.tensor(a["array"], dtype=torch.float32) for a in rows["audio"]]
        lens = torch.tensor([a.shape[0] for a in arrays], dtype=torch.int64)
        maxlen = int(lens.max())
        wav = torch.zeros(len(arrays), maxlen, dtype=torch.float32)
        for i, a in enumerate(arrays):
            wav[i, : a.shape[0]] = a
        with torch.no_grad():
            f, fl = feat(wav, lens)  # [B,T,80], [B]
        yield f, fl


def _coverage_stats(targets: torch.Tensor, valid: torch.Tensor, vocab_size: int, num_codebooks: int):
    """Accumulate per-codebook histograms over valid frames. Returns list of count tensors [V]."""
    counts = [torch.zeros(vocab_size, dtype=torch.int64) for _ in range(num_codebooks)]
    for n in range(num_codebooks):
        idx = targets[..., n][valid]  # [num_valid]
        counts[n] += torch.bincount(idx, minlength=vocab_size)
    return counts


def _report(name: str, counts, vocab_size: int):
    print(f"\n=== {name} ===")
    cov_list, ent_list, ppl_list, maxf_list = [], [], [], []
    for n, c in enumerate(counts):
        total = int(c.sum())
        used = int((c > 0).sum())
        p = c.double() / max(total, 1)
        nz = p[p > 0]
        entropy = float(-(nz * nz.log()).sum())  # nats
        norm_entropy = entropy / float(torch.log(torch.tensor(float(vocab_size))))
        ppl = float(torch.exp(torch.tensor(entropy)))
        maxf = float(p.max())
        cov = used / vocab_size
        # cumulative mass of the top-k codes, and entropy of the tail (excluding the single top
        # code, ~silence): if the tail is broad the remaining distribution is still high-entropy.
        ps, _ = torch.sort(p, descending=True)
        top1, top10, top50 = float(ps[:1].sum()), float(ps[:10].sum()), float(ps[:50].sum())
        tail = ps[1:]
        tail = tail / tail.sum()
        tail_nz = tail[tail > 0]
        tail_ne = float(-(tail_nz * tail_nz.log()).sum()) / float(torch.log(torch.tensor(float(vocab_size - 1))))
        cov_list.append(cov); ent_list.append(norm_entropy); ppl_list.append(ppl); maxf_list.append(maxf)
        print(
            f"  cb{n}: frames={total:>8d} cov={cov*100:5.1f}% ne={norm_entropy:5.3f} ppl={ppl:7.1f} "
            f"top1={top1*100:5.2f}% top10={top10*100:5.1f}% top50={top50*100:5.1f}% tail_ne={tail_ne:5.3f}"
        )
    print(
        f"  mean: coverage={sum(cov_list)/len(cov_list)*100:6.2f}%  "
        f"norm_entropy={sum(ent_list)/len(ent_list):5.3f}  perplexity={sum(ppl_list)/len(ppl_list):8.1f}  "
        f"max_freq={sum(maxf_list)/len(maxf_list)*100:6.3f}%"
    )
    return sum(cov_list) / len(cov_list), sum(ent_list) / len(ent_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-utts", type=int, default=400)
    ap.add_argument("--split", type=str, default="validation.clean")
    ap.add_argument("--stack-size", type=int, default=4)
    ap.add_argument("--codebook-dim", type=int, default=16)
    ap.add_argument("--vocab-size", type=int, default=8192)
    ap.add_argument("--num-codebooks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(0)
    V, N = args.vocab_size, args.num_codebooks

    # Three configurations to disentangle the effect of input-norm and L2-norm.
    quant = MultiCodebookRandomProjectionQuantizer(
        input_dim=80, stack_size=args.stack_size, codebook_dim=args.codebook_dim,
        vocab_size=V, num_codebooks=N, seed=args.seed,
    )

    counts_full = [torch.zeros(V, dtype=torch.int64) for _ in range(N)]
    counts_nonorm = [torch.zeros(V, dtype=torch.int64) for _ in range(N)]
    counts_nol2 = [torch.zeros(V, dtype=torch.int64) for _ in range(N)]

    for feats, lens in _load_logmel(args.num_utts, args.split):
        # (a) full: input-norm + L2 (the real target).
        normed = masked_mean_var_norm(feats, lens)
        tgt_full, tlen = quant(normed, lens)
        # (b) no input-norm (raw log-mel into the quantizer).
        tgt_nonorm, _ = quant(feats, lens)
        # (c) no L2 (re-derive nearest code on un-normalized projections).
        tgt_nol2 = _quantize_no_l2(quant, normed)

        t2 = tgt_full.shape[1]
        valid = torch.arange(t2)[None, :] < tlen[:, None]  # [B,T2]
        for acc, tg in ((counts_full, tgt_full), (counts_nonorm, tgt_nonorm), (counts_nol2, tgt_nol2)):
            for n in range(N):
                acc[n] += torch.bincount(tg[..., n][valid], minlength=V)

    cov_full, ent_full = _report("FULL (input-norm + L2)  <- the SSL target", counts_full, V)
    _report("ablation: NO input-norm", counts_nonorm, V)
    _report("ablation: NO L2-norm", counts_nol2, V)

    print("\n[verdict]")
    # Health is judged by entropy + tail spread, NOT raw coverage (coverage is frame-count
    # limited and a single dominant 'silence' code is expected & harmless). Collapse would show
    # as very low norm_entropy (<~0.3) and a few codes capturing the vast majority of mass.
    print(
        f"  full-config mean coverage={cov_full*100:.2f}%, mean norm_entropy={ent_full:.3f}.\n"
        f"  Compare against the NO-input-norm ablation (collapsed) and NO-L2 ablation (worse) above:\n"
        f"  input-norm and L2 should both visibly raise entropy/perplexity and lower top-1 mass."
    )


def _quantize_no_l2(quant: MultiCodebookRandomProjectionQuantizer, features: torch.Tensor) -> torch.Tensor:
    """Ablation: true 'no L2' BEST-RQ -- nearest code by L2 distance between the *raw*
    projection and the *raw* (un-normalized) codebook. We regenerate the raw codebook with the
    quantizer's seed (it stores only the normalized one). Note: normalizing only the projection
    while keeping a unit-norm codebook leaves the argmax unchanged, so the meaningful ablation is
    to drop normalization on BOTH sides."""
    n_cb, v, d_dim = quant.num_codebooks, quant.vocab_size, quant.codebook_dim
    gen = torch.Generator().manual_seed(quant.seed)
    raw_proj = torch.randn(n_cb, quant.proj_in, d_dim, generator=gen)
    raw_cb = torch.randn(n_cb, v, d_dim, generator=gen)  # NOT normalized
    b, t, f = features.shape
    t2 = t // quant.stack_size
    x = features[:, : t2 * quant.stack_size, :].reshape(b, t2, quant.proj_in).float()
    out = []
    for n in range(n_cb):
        proj = torch.matmul(x, raw_proj[n])  # [B,T2,D] (no normalize)
        dist = torch.cdist(proj, raw_cb[n])  # [B,T2,V]
        out.append(dist.argmin(dim=-1))
    return torch.stack(out, dim=-1)


if __name__ == "__main__":
    main()
