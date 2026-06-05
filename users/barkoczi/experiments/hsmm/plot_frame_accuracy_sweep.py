#!/usr/bin/env python3
"""Plot frame accuracy heatmaps for the wav2vec2-large PCA sweep."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import numpy as np


DEFAULT_ROOT = (
    "alias/example_setups/librispeech/feature_dump/"
    "ls960_wav2vec2_large_pca_label_stats_sweep"
)


def iter_json_files(root: Path):
    pattern = "wav2vec2_large_layer_*/pca_*/*_accuracy/frame_accuracy_job/output/*_frame_accuracy.json"
    yield from root.glob(pattern)

    single_run_pattern = "*_accuracy/frame_accuracy_job/output/*_frame_accuracy.json"
    yield from root.glob(single_run_pattern)

    if root.name == "output":
        yield from root.glob("*_frame_accuracy.json")


def progress(current: int, total: int, label: str) -> None:
    if total <= 0:
        return
    width = 32
    filled = round(width * current / total)
    bar = "#" * filled + "-" * (width - filled)
    percent = 100.0 * current / total
    end = "\n" if current >= total else "\r"
    print(f"{label}: [{bar}] {current}/{total} ({percent:5.1f}%)", end=end, file=sys.stderr, flush=True)


def parse_result(path: Path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    path_text = str(path)

    layer_match = re.search(r"wav2vec2_large_layer_(\d+)", path_text)
    pca_match = re.search(r"pca_(\d+)", path_text)
    mode_match = re.search(r"/(framewise|viterbi)_accuracy(?:/|$)", path_text)

    if not layer_match:
        layer_match = re.search(r"layer_(\d+)_pca_\d+_", path.name)
    if not pca_match:
        pca_match = re.search(r"layer_\d+_pca_(\d+)_", path.name)
    if not mode_match:
        mode_match = re.search(r"_(framewise|viterbi)_frame_accuracy\.json$", path.name)

    decode_mode = mode_match.group(1) if mode_match else payload.get("decode_mode")
    if not layer_match or not pca_match or decode_mode not in {"framewise", "viterbi"}:
        return None

    return {
        "decode_mode": decode_mode,
        "layer": int(layer_match.group(1)),
        "pca_dim": int(pca_match.group(1)),
        "accuracy": float(payload["accuracy"]),
        "correct_frames": int(payload.get("correct_frames", 0)),
        "total_frames": int(payload.get("total_frames", 0)),
        "num_sequences": int(payload.get("num_sequences", 0)),
        "path": str(path),
    }


def read_results(root: Path):
    rows = []
    paths = list(iter_json_files(root))
    progress(0, len(paths), "reading JSON")
    for idx, path in enumerate(paths, start=1):
        try:
            row = parse_result(path)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            continue
        if row is not None:
            rows.append(row)
        progress(idx, len(paths), "reading JSON")
    return sorted(rows, key=lambda r: (r["decode_mode"], r["layer"], r["pca_dim"]))


def write_csv(rows, path: Path) -> None:
    print(f"writing CSV: {path}", file=sys.stderr)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ["decode_mode", "layer", "pca_dim", "accuracy", "correct_frames", "total_frames", "num_sequences", "path"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in columns})


def make_matrix(rows, mode, layers, pca_dims):
    values = np.full((len(layers), len(pca_dims)), np.nan)
    layer_to_idx = {layer: i for i, layer in enumerate(layers)}
    pca_to_idx = {pca_dim: i for i, pca_dim in enumerate(pca_dims)}
    for row in rows:
        if row["decode_mode"] == mode:
            values[layer_to_idx[row["layer"]], pca_to_idx[row["pca_dim"]]] = row["accuracy"]
    return values


def annotate(ax, values, *, diff=False):
    finite = values[np.isfinite(values)]
    midpoint = float(np.nanmean(finite)) if finite.size else 0.0
    for y in range(values.shape[0]):
        for x in range(values.shape[1]):
            value = values[y, x]
            if not np.isfinite(value):
                text, color = "n/a", "black"
            elif diff:
                text, color = f"{value * 100:+.2f}", "black"
            else:
                text = f"{value * 100:.2f}"
                color = "white" if value > midpoint else "black"
            ax.text(x, y, text, ha="center", va="center", fontsize=9, color=color)


def plot(rows, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    progress(0, 4, "plotting")
    layers = sorted({row["layer"] for row in rows})
    pca_dims = sorted({row["pca_dim"] for row in rows})
    framewise = make_matrix(rows, "framewise", layers, pca_dims)
    viterbi = make_matrix(rows, "viterbi", layers, pca_dims)
    diff = viterbi - framewise
    progress(1, 4, "plotting")

    panels = [("Framewise", framewise, False), ("Viterbi", viterbi, False)]
    if np.isfinite(diff).any():
        panels.append(("Viterbi - framewise", diff, True))

    fig, axes = plt.subplots(1, len(panels), figsize=(5.0 * len(panels), 4.0), constrained_layout=True)
    if len(panels) == 1:
        axes = [axes]
    progress(2, 4, "plotting")

    acc_values = np.concatenate([framewise[np.isfinite(framewise)], viterbi[np.isfinite(viterbi)]])
    acc_vmin = float(acc_values.min()) if acc_values.size else 0.0
    acc_vmax = float(acc_values.max()) if acc_values.size else 1.0

    for ax, (title, values, is_diff) in zip(axes, panels):
        if is_diff:
            limit = max(float(np.nanmax(np.abs(values))) if np.isfinite(values).any() else 0.01, 0.001)
            image = ax.imshow(values, cmap="coolwarm", vmin=-limit, vmax=limit)
        else:
            image = ax.imshow(values, cmap="viridis", vmin=acc_vmin, vmax=acc_vmax)

        ax.set_title(title)
        ax.set_xlabel("PCA dim")
        ax.set_ylabel("Layer")
        ax.set_xticks(np.arange(len(pca_dims)), [str(x) for x in pca_dims])
        ax.set_yticks(np.arange(len(layers)), [str(y) for y in layers])
        annotate(ax, values, diff=is_diff)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("accuracy" if not is_diff else "accuracy difference")
    progress(3, 4, "plotting")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    progress(4, 4, "plotting")


def print_summary(rows) -> None:
    for mode in ["framewise", "viterbi"]:
        mode_rows = [row for row in rows if row["decode_mode"] == mode]
        if not mode_rows:
            continue
        best = max(mode_rows, key=lambda row: row["accuracy"])
        print(
            f"{mode}: best layer={best['layer']} pca_dim={best['pca_dim']} "
            f"accuracy={best['accuracy'] * 100:.3f}% frames={best['total_frames']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument("--out", type=Path, default=Path("frame_accuracy_sweep_heatmap.png"))
    parser.add_argument("--csv", type=Path, default=Path("frame_accuracy_sweep.csv"))
    args = parser.parse_args()

    rows = read_results(args.root)
    if not rows:
        raise SystemExit(f"No frame accuracy JSON files found below {args.root}")

    write_csv(rows, args.csv)
    print_summary(rows)
    plot(rows, args.out)
    print(f"wrote {args.csv}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
