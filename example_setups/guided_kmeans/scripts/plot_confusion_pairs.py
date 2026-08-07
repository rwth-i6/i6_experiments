
INPUT_GLOB = (
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/guided_kmeans"
    "/testing_experimental/confusion_pairs"
    "/fb_lm-3gram-1.0-transition-1.0_loop-0.7-sil-loop-0.7_ls-100_cheating_epoch-*_frame_confusion"
)

OUT_DIR = (
    "/u/lkleppel/experiments/20260520_unsupervised_asr/output/guided_kmeans"
    "/fb_test/confusion_pairs_plots"
)

TOP_K  = 15  # number of pairs shown in the trend chart
NCOLS  = 4   # columns in the heatmap grid


import csv
import glob
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# helpers

def _epoch_from_path(path: str) -> int:
    # Extract the epoch number from a filename like *_epoch-3_*
    m = re.search(r"_epoch[-_](\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    raise ValueError(
        f"Cannot extract epoch from '{path}'. "
        "Either rename files to contain '_epoch-N_' or pass files in order."
    )


def _load_tsv(path: str) -> list[dict]:
    # Load a confusion TSV as a list of {ref, hyp, count} dicts
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append({"ref": row["ref"].strip(), "hyp": row["hyp"].strip(), "count": int(row["count"])})
    return rows


def load_files(paths: list[str]) -> dict[int, list[dict]]:
    # Return {epoch: [{ref, hyp, count}, ...]} sorted by epoch
    data: dict[int, list[dict]] = {}
    for p in paths:
        try:
            epoch = _epoch_from_path(p)
        except ValueError:
            epoch = paths.index(p)
        data[epoch] = _load_tsv(p)
    return dict(sorted(data.items()))


# plot 1: top-K error pair trends

def plot_top_k_trends(
    epoch_data: dict[int, list[dict]],
    top_k: int = 15,
    out_path: str = "confusion_trends.png",
):
    epochs = sorted(epoch_data.keys())

    # Aggregate total error count per (ref, hyp) pair across all epochs
    pair_totals: dict[tuple, int] = defaultdict(int)
    for rows in epoch_data.values():
        for row in rows:
            if row["ref"] != row["hyp"]:
                pair_totals[(row["ref"], row["hyp"])] += row["count"]

    top_pairs = sorted(pair_totals, key=lambda p: -pair_totals[p])[:top_k]

    # Build (epoch x pair) count matrix
    counts: dict[tuple, list] = {pair: [] for pair in top_pairs}
    for ep in epochs:
        lookup = {(r["ref"], r["hyp"]): r["count"] for r in epoch_data[ep]}
        for pair in top_pairs:
            counts[pair].append(lookup.get(pair, 0))

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.get_cmap("tab20")
    for i, pair in enumerate(top_pairs):
        label = f"{pair[0]} → {pair[1]}"
        ax.plot(epochs, counts[pair], marker="o", label=label, color=cmap(i / top_k), linewidth=1.5, markersize=4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Substitution count")
    ax.set_title(f"Top-{top_k} confusion pairs over training epochs")
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# plot 2: confusion matrix heatmap grid

def _build_matrix(rows: list[dict], phonemes: list[str]) -> np.ndarray:
    # Row = ref, col = hyp. Values = normalized by row sum
    idx = {p: i for i, p in enumerate(phonemes)}
    n = len(phonemes)
    mat = np.zeros((n, n), dtype=float)
    for row in rows:
        r, h = row["ref"], row["hyp"]
        if r in idx and h in idx:
            mat[idx[r], idx[h]] += row["count"]
    # Normalize each row by its total (so diagonal = recall per phoneme)
    row_sums = mat.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    return mat / row_sums


def plot_confusion_matrices(
    epoch_data: dict[int, list[dict]],
    out_path: str = "confusion_matrix.png",
    ncols: int = 4,
):
    epochs = sorted(epoch_data.keys())
    n_epochs = len(epochs)
    nrows = max(1, (n_epochs + ncols - 1) // ncols)

    # Collect all phoneme labels present across all epochs
    all_phonemes: set[str] = set()
    for rows in epoch_data.values():
        for row in rows:
            all_phonemes.add(row["ref"])
            all_phonemes.add(row["hyp"])
    # Sort: silence first, then alphabetical
    phonemes = sorted(all_phonemes, key=lambda p: ("" if p.startswith("[") else "~") + p)

    tick_step = max(1, len(phonemes) // 20)  # avoid overcrowding if many phonemes

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5))
    axes = np.array(axes).reshape(nrows, ncols)

    vmax = 1.0
    for i, ep in enumerate(epochs):
        row, col = divmod(i, ncols)
        ax = axes[row, col]
        mat = _build_matrix(epoch_data[ep], phonemes)
        im = ax.imshow(mat, aspect="auto", cmap="Blues", vmin=0, vmax=vmax)
        ax.set_title(f"Epoch {ep}", fontsize=10)
        ticks = list(range(0, len(phonemes), tick_step))
        labels = [phonemes[t] for t in ticks]
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels, fontsize=6)
        if row == nrows - 1:
            ax.set_xlabel("Predicted", fontsize=8)
        if col == 0:
            ax.set_ylabel("True", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Hide unused subplots
    for j in range(n_epochs, nrows * ncols):
        r, c = divmod(j, ncols)
        axes[r, c].set_visible(False)

    fig.suptitle("Confusion matrices per epoch (row-normalized)", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")




if __name__ == "__main__":
    paths = sorted(glob.glob(INPUT_GLOB))
    if not paths:
        sys.exit(f"No files matched: {INPUT_GLOB}")

    print(f"Loading {len(paths)} file(s)...")
    epoch_data = load_files(paths)
    print(f"Epochs found: {sorted(epoch_data.keys())}")

    os.makedirs(OUT_DIR, exist_ok=True)

    plot_top_k_trends(
        epoch_data,
        top_k=TOP_K,
        out_path=os.path.join(OUT_DIR, "confusion_trends.png"),
    )
    plot_confusion_matrices(
        epoch_data,
        out_path=os.path.join(OUT_DIR, "confusion_matrix.png"),
        ncols=NCOLS,
    )
