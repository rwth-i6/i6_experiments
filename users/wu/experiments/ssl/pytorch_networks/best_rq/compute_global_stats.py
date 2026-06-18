"""
Compute the global (corpus-level) per-feature log-mel mean/std over LibriSpeech 960h and write
``ls_logmel_stats.py``. Samples utterances UNIFORMLY across the pooled three 960h train splits
(clean-100 + clean-360 + other-500), so the average is weighted by data amount (i.e. dominated by
clean-360 + other-500, including the noisier 'other' condition) -- the true training distribution.

Run (login node, offline):
    cd /e/project1/spell/wu24/2026-06-17_ssl
    HF_HOME=/e/project1/spell/common_hf_home PYTHONPATH=recipe:recipe/i6_models \
      /e/project1/spell/wu24/env/conda/envs/speech_llm/bin/python \
      recipe/i6_experiments/users/wu/experiments/ssl/pytorch_networks/best_rq/compute_global_stats.py \
      --num-utts 10000
"""

from __future__ import annotations

import argparse
import glob
import os
import random

import torch


TRAIN_960H_SPLITS = ("train.clean.100", "train.clean.360", "train.other.500")


def _parquet_files(splits):
    hf = os.environ.get("HF_HOME", "/e/project1/spell/common_hf_home")
    repo = os.path.join(hf, "hub", "datasets--openslr--librispeech_asr")
    commit = open(os.path.join(repo, "refs", "main")).read().strip()
    files = []
    for s in splits:
        files += sorted(glob.glob(os.path.join(repo, "snapshots", commit, "all", s, "*.parquet")))
    assert files
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-utts", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import datasets
    from i6_models.primitives.feature_extraction import (
        LogMelFeatureExtractionV1,
        LogMelFeatureExtractionV1Config,
    )

    feat = LogMelFeatureExtractionV1(
        cfg=LogMelFeatureExtractionV1Config(
            sample_rate=16000, win_size=0.025, hop_size=0.01, f_min=60, f_max=7600,
            min_amp=1e-10, num_filters=80, center=False,
        )
    ).eval()

    files = _parquet_files(TRAIN_960H_SPLITS)
    ds = datasets.load_dataset("parquet", data_files={"x": files}, split="x")
    ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16000, decode=True))
    rng = random.Random(args.seed)
    n = min(args.num_utts, len(ds))
    idxs = sorted(rng.sample(range(len(ds)), n))
    print(f"[stats] pooled 960h utts={len(ds)}, sampling {n} (seed {args.seed})")

    s = torch.zeros(80, dtype=torch.float64)
    ss = torch.zeros(80, dtype=torch.float64)
    cnt = 0
    B = 32
    for b0 in range(0, n, B):
        rows = ds[idxs[b0 : b0 + B]]
        arrs = [torch.tensor(a["array"], dtype=torch.float32) for a in rows["audio"]]
        ln = torch.tensor([a.shape[0] for a in arrs])
        mx = int(ln.max())
        wav = torch.zeros(len(arrs), mx)
        for i, a in enumerate(arrs):
            wav[i, : a.shape[0]] = a
        with torch.no_grad():
            f, fl = feat(wav, ln)
        for i, m in enumerate(fl.tolist()):
            v = f[i, :m].double()
            s += v.sum(0)
            ss += (v * v).sum(0)
            cnt += m
        if (b0 // B) % 50 == 0:
            print(f"  ...{b0 + len(arrs)}/{n} utts, {cnt} frames", flush=True)

    mean = (s / cnt).float()
    std = torch.sqrt((ss / cnt).float() - mean ** 2 + 1e-5)
    print(f"[stats] done: {n} utts / {cnt} frames; mean[:5]={mean[:5].tolist()} std[:5]={std[:5].tolist()}")

    out = os.path.join(os.path.dirname(__file__), "ls_logmel_stats.py")
    ml = ", ".join(f"{x:.6f}" for x in mean.tolist())
    sl = ", ".join(f"{x:.6f}" for x in std.tolist())
    with open(out, "w") as fo:
        fo.write('"""\n')
        fo.write("Global (corpus-level) per-feature log-mel mean/std for LibriSpeech 960h, used as the\n")
        fo.write("FIXED, sequence-independent input normalization for BEST-RQ.\n\n")
        fo.write(f"Computed over {n} utterances ({cnt} frames) sampled UNIFORMLY across the pooled 960h\n")
        fo.write("train splits (clean-100 + clean-360 + other-500), with LogMelFeatureExtractionV1Config(\n")
        fo.write("sample_rate=16000, win_size=0.025, hop_size=0.01, f_min=60, f_max=7600, min_amp=1e-10,\n")
        fo.write("num_filters=80, center=False). Regenerate via compute_global_stats.py.\n")
        fo.write('"""\n\n')
        fo.write("LOGMEL_MEAN = [" + ml + "]\n\n")
        fo.write("LOGMEL_STD = [" + sl + "]\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
