"""
Analyze centroid evolution across epochs from a GuidedKMeans training run

Reads centroids.*.npy and epoch_statistics.json from OUTPUT_DIR
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


OUTPUT_DIR = Path(
    "/u/lkleppel/experiments/20260520_unsupervised_asr/work/i6_core/returnn/forward/ReturnnForwardJobV2.5yp9qWxgVP5l/output"
)

# RASR phoneme order as used by the decoder
PHONEME_NAMES = [
    "[SILENCE]", "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D",
    "DH", "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH", "T", "TH",
    "UH", "UW", "V", "W", "Y", "Z", "ZH", #"[UNKNOWN]",
]


def cosine_matrix(c: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(c, axis=1, keepdims=True)
    valid = (norms > 0).flatten()
    cn = np.where(norms > 0, c / np.where(norms > 0, norms, 1), 0.0)
    cos = cn @ cn.T
    # mask zero-norm rows/cols
    cos[~valid, :] = np.nan
    cos[:, ~valid] = np.nan
    np.fill_diagonal(cos, np.nan)
    return cos


def top_pairs(cos: np.ndarray, names: list[str], n: int = 5) -> list[tuple]:
    flat = [(cos[i, j], names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if not np.isnan(cos[i, j])]
    flat.sort(reverse=True)
    return flat[:n]


def bottom_pairs(cos: np.ndarray, names: list[str], n: int = 5) -> list[tuple]:
    flat = [(cos[i, j], names[i], names[j])
            for i in range(len(names))
            for j in range(i + 1, len(names))
            if not np.isnan(cos[i, j])]
    flat.sort()
    return flat[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", nargs="*", default=None,
                        help="Phoneme pairs to track, e.g. AH:DH D:N")
    args = parser.parse_args()

    out = OUTPUT_DIR
    centroid_files = sorted(out.glob("centroids.*.npy"),
                            key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not centroid_files:
        raise FileNotFoundError(f"No centroids.*.npy found in {out}")

    names = PHONEME_NAMES

    # Load epoch statistics if available
    stats_path = out / "epoch_statistics.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)

    # Parse tracked pairs
    tracked_pairs = []
    if args.pairs:
        for spec in args.pairs:
            a, b = spec.split(":")
            tracked_pairs.append((a.strip(), b.strip()))
    else:
        # Default: most-discussed pairs
        tracked_pairs = [("AH", "DH"), ("D", "N")]

    pair_indices = []
    for a, b in tracked_pairs:
        if a not in names or b not in names:
            print(f"Warning: phoneme {a!r} or {b!r} not in phoneme list, skipping pair")
            continue
        pair_indices.append((names.index(a), names.index(b), a, b))

    # Header
    col_w = [max(len(a) + 1 + len(b), 9) for _, _, a, b in pair_indices]
    pair_header = "  ".join(f"{a+'-'+b:>{w}}" for (_, _, a, b), w in zip(pair_indices, col_w))
    base_header = f"{'Epoch':>6}  {'Mean cos':>9}  {'Max cos':>9}  {'Min cos':>9}"
    print(base_header + (f"  {pair_header}" if pair_indices else ""))
    print("-" * (40 + sum(w + 2 for w in col_w)))

    all_cos = {}

    for cf in centroid_files:
        epoch = int(re.search(r"(\d+)", cf.stem).group(1))
        c = np.load(cf)
        cos = cosine_matrix(c)
        all_cos[epoch] = cos

        mean_c = float(np.nanmean(cos))
        max_c = float(np.nanmax(cos))
        min_c = float(np.nanmin(cos))

        row = f"{epoch:>6}  {mean_c:>9.6f}  {max_c:>9.6f}  {min_c:>9.6f}"
        for (ia, ib, a, b), w in zip(pair_indices, col_w):
            v = cos[ia, ib]
            row += f"  {v:>{w}.6f}"
        print(row)

    # Per-epoch: most / least similar pairs
    print()
    for epoch in sorted(all_cos):
        cos = all_cos[epoch]
        ep_label = f"Epoch {epoch}"
        top = top_pairs(cos, names, n=3)
        bot = bottom_pairs(cos, names, n=3)
        top_str = ", ".join(f"{a}-{b} ({v:.4f})" for v, a, b in top)
        bot_str = ", ".join(f"{a}-{b} ({v:.4f})" for v, a, b in bot)
        print(f"{ep_label:>8}  most similar: {top_str}")
        print(f"          least similar: {bot_str}")

    # Per-epoch: score / duration / loop frequency
    if stats:
        print()
        print(f"{'Epoch':>6}  {'Avg total score':>16}  {'Avg normed score':>16}  {'Avg seg dur':>12}  {'Loop freq':>10}")
        print("-" * 70)
        for epoch_key in sorted(stats.keys(), key=int):
            ep = stats[epoch_key]
            total  = ep.get("average_total_score",        float("nan"))
            normed = ep.get("average_total_normed_score", float("nan"))
            dur    = ep.get("average_segment_duration",   float("nan"))
            loop   = ep.get("relative_loop_frequency",    float("nan"))
            print(f"{int(epoch_key):>6}  {total:>16.1f}  {normed:>16.1f}  {dur:>12.4f}  {loop:>10.4f}")

    # Per-epoch: phoneme frequency from epoch_statistics.json
    if stats:
        print()
        print(f"{'Epoch':>6}  {'Fraction visited':>18}  {'Top 5 phonemes by count'}")
        print("-" * 70)
        for epoch_key in sorted(stats.keys(), key=int):
            ep = stats[epoch_key]
            frac = ep.get("fraction_visited_phonemes", float("nan"))
            counts = ep.get("absolute_phoneme_counts", {})
            top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            top5_str = "  ".join(f"{ph}={cnt}" for ph, cnt in top5)
            print(f"{int(epoch_key):>6}  {frac:>18.4f}  {top5_str}")


if __name__ == "__main__":
    main()
