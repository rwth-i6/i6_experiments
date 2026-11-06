"""Compare blank minus best-nonblank logits at each saved recognition call.

Pass post-output-quantization recordings. Blank is the last output channel.
Positive margins mean blank strictly wins; zero means a top-score tie.
These are each system's own search calls, not paired identical-input results.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_histograms import iter_records


def blank_margins(logits):
    if logits.ndim != 2 or logits.shape[1] < 4 or not np.isfinite(logits).all():
        raise ValueError("Expected finite [calls, vocabulary] logits with at least four channels")
    blank = logits[:, -1]
    return {
        "all_nonblank": blank - logits[:, :-1].max(axis=1),
        "ordinary_tokens": blank - logits[:, 2:-1].max(axis=1),
    }


def collect(path):
    parts = {"all_nonblank": [], "ordinary_tokens": []}
    records = nonempty_records = 0
    for record in iter_records(path):
        records += 1
        batches = record["linear2"]
        if not batches:
            continue
        nonempty_records += 1
        margins = blank_margins(np.concatenate(batches, axis=0))
        for name in parts:
            parts[name].append(margins[name])
    if not parts["all_nonblank"]:
        raise ValueError(f"No layer-2 outputs in {path}")
    values = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    return values, {"path": str(path.resolve()), "records": records,
                    "nonempty_records": nonempty_records}


def summarize(values):
    percentiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "calls": int(values.size),
        "mean": float(np.mean(values, dtype=np.float64)),
        "std": float(np.std(values, dtype=np.float64)),
        "min": float(values.min()), "max": float(values.max()),
        "blank_strict_win_fraction": float(np.mean(values > 0)),
        "top_score_tie_fraction": float(np.mean(values == 0)),
        "nonblank_strict_win_fraction": float(np.mean(values < 0)),
        "percentiles": dict(zip(["p1", "p5", "p25", "p50", "p75", "p95", "p99"], percentiles.tolist())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "blank_margins_post_quant")
    args = parser.parse_args()
    if len(args.input) != len(args.labels) or args.bins < 1:
        parser.error("Provide one label per input and a positive bin count")
    data, reports = [], []
    for path, label in zip(args.input, args.labels):
        print(f"Reading {label}: {path}", flush=True)
        values, metadata = collect(path)
        data.append(values)
        reports.append({"label": label, **metadata,
                        "margins": {name: summarize(v) for name, v in values.items()}})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for name in data[0]:
        low = min(d[name].min() for d in data)
        high = max(d[name].max() for d in data)
        if low == high:
            low, high = low - 0.5, high + 0.5
        edges = np.linspace(low, high, args.bins + 1)
        fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for i, (values, label) in enumerate(zip(data, args.labels)):
            counts, _ = np.histogram(values[name], bins=edges)
            assert int(counts.sum()) == values[name].size
            ax.stairs(counts / (values[name].size * np.diff(edges)), edges,
                      fill=True, alpha=0.65, color=["#447fb8", "#edb637"][i % 2], label=label)
            arrays[f"input_{i}_{name}"] = values[name]
        ax.axvline(0, color="black", linewidth=1, linestyle="--")
        ax.set_xlabel("Blank logit − best nonblank logit (after output quantization)")
        ax.set_ylabel("Probability density")
        qualifier = "All nonblank tokens" if name == "all_nonblank" else "Excluding </s> and <unk> from competitors"
        ax.set_title(f"Blank advantage during greedy recognition · dev-other\n{qualifier}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        for extension in ("png", "pdf"):
            fig.savefig(args.output_dir / f"blank_margin_{name}.{extension}", dpi=180)
        plt.close(fig)
    report = {"measurement": "post-output-quantization", "blank_channel": "last",
              "comparison": "separate search calls, not matched inputs",
              "inputs": reports}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "per_call_margins.npz", **arrays)
    print(json.dumps(report, indent=2))
    print(f"Saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
