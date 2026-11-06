"""Plot search-time joiner outputs from appended np.save dictionary records.

Run from the QAT setup root, for example:
    python diagnostics/qat_ffnn_joiner_activations/plot_histograms.py \
        --input path/to/joint_activations_debug.npy --labels 'Memristor, ADC 4/4'

Multiple --input paths with corresponding --labels produce overlaid histograms
with common bins. All scalar activations are counted; no subsampling is used.
Only one saved record is loaded at a time. These are complete linear-module
outputs, before output activation quantization. Use --layers predictor_linear1
to plot the first prediction-network linear layer from predictor debug recordings.
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYERS = {
    "linear1": "Joiner linear 1 output (pre-tanh)",
    "linear2": "Joiner linear 2 output (pre-output-quantization logit)",
}


def iter_records(path):
    with path.open("rb") as stream:
        file_size = os.fstat(stream.fileno()).st_size
        while stream.tell() < file_size:
            yield np.load(stream, allow_pickle=True).item()


def layer_values(record, layer, exclude_special_tokens=False):
    batches = record[layer]
    if exclude_special_tokens and layer == "linear2":
        # Vocabulary order: </s>, <unk>, ordinary BPE tokens, blank.
        batches = [batch[:, 2:-1] for batch in batches]
    return np.concatenate([batch.ravel() for batch in batches]) if batches else np.empty(0)


def scan_statistics(path, exclude_special_tokens=False):
    stats = {
        layer: {
            "count": 0,
            "nonfinite": 0,
            "min": float("inf"),
            "max": -float("inf"),
            "mean": 0.0,
            "m2": 0.0,
            "batches": 0,
            "rows": 0,
            "widths": set(),
        }
        for layer in LAYERS
    }
    records = empty_records = 0
    for record in iter_records(path):
        records += 1
        if all(not record[layer] for layer in LAYERS):
            empty_records += 1
        for layer, s in stats.items():
            values = layer_values(record, layer, exclude_special_tokens)
            finite = np.isfinite(values)
            s["nonfinite"] += int(values.size - finite.sum())
            values = values[finite].astype(np.float64)
            for batch in record[layer]:
                s["batches"] += 1
                s["rows"] += batch.shape[0]
                s["widths"].add(batch.shape[1] - (3 if exclude_special_tokens and layer == "linear2" else 0))
            if not values.size:
                continue
            n = values.size
            mean = float(values.mean())
            delta = mean - s["mean"]
            total = s["count"] + n
            s["m2"] += float(np.sum((values - mean) ** 2)) + delta ** 2 * s["count"] * n / total
            s["mean"] += delta * n / total
            s["count"] = total
            s["min"] = min(s["min"], float(values.min()))
            s["max"] = max(s["max"], float(values.max()))
    for layer, s in stats.items():
        if not s["count"]:
            raise ValueError(f"No finite outputs for {layer} in {path}")
        s["std"] = float(np.sqrt(s.pop("m2") / s["count"]))
        s["widths"] = sorted(s["widths"])
    return {"path": str(path.resolve()), "records": records, "empty_records": empty_records, "layers": stats}


def count_histograms(path, edges, exclude_special_tokens=False):
    counts = {layer: np.zeros(len(bins) - 1, dtype=np.int64) for layer, bins in edges.items()}
    for record in iter_records(path):
        for layer in LAYERS:
            values = layer_values(record, layer, exclude_special_tokens)
            counts[layer] += np.histogram(values[np.isfinite(values)], bins=edges[layer])[0]
    return counts


def draw_histogram(ax, layer, edges, histograms, labels, normalization, exclude_special_tokens=False):
    colors = ["#447fb8", "#edb637"] if len(labels) > 1 else ["#edb637"]
    for index, (values, label) in enumerate(zip(histograms, labels)):
        ax.stairs(values[layer], edges[layer], fill=True, alpha=0.65,
                  color=colors[index % len(colors)], label=label, linewidth=0.8)
    ax.set_xlabel(LAYERS[layer])
    if exclude_special_tokens and layer == "linear2":
        ax.set_title("Ordinary tokens only (excluding </s>, <unk>, blank)", fontsize=10)
    ax.set_ylabel("Probability density" if normalization == "density" else "Frequency")
    ax.spines[["top", "right"]].set_visible(False)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4))
    ax.legend(frameon=False)


def main():
    global LAYERS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", nargs="+", type=Path, required=True)
    layer_labels = {**LAYERS, "predictor_linear1": "Prediction network linear 1 output"}
    parser.add_argument("--layers", nargs="+", choices=list(layer_labels),
                        default=list(LAYERS), help="Saved activation layers to plot")
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--bins", type=int, default=100)
    parser.add_argument("--exclude-special-tokens", action="store_true")
    parser.add_argument("--post-output-quantization", action="store_true")
    parser.add_argument("--normalization", choices=["density", "count"], default="density")
    parser.add_argument("--hide-ticks", action="store_true")
    parser.add_argument("--hide-y-ticks", action="store_true")
    parser.add_argument("--shared-x-axis", action="store_true")
    parser.add_argument("--shared-y-axis", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "plots")
    parser.add_argument("--title", default="Joiner outputs during greedy recognition · dev-other")
    args = parser.parse_args()
    LAYERS = {layer: layer_labels[layer] for layer in args.layers}
    if args.post_output_quantization:
        LAYERS["linear2"] = "Joiner logits (after output quantization)"
    labels = args.labels or [path.parent.name for path in args.input]
    if len(labels) != len(args.input) or args.bins < 1:
        parser.error("Provide one label per input and a positive bin count")

    summaries = []
    for path, label in zip(args.input, labels):
        print(f"Scanning {label}: {path}", flush=True)
        summaries.append(scan_statistics(path, args.exclude_special_tokens))
    edges = {}
    for layer in LAYERS:
        low = min(summary["layers"][layer]["min"] for summary in summaries)
        high = max(summary["layers"][layer]["max"] for summary in summaries)
        if low == high:
            low, high = low - 0.5, high + 0.5
        edges[layer] = np.linspace(low, high, args.bins + 1)
    if args.shared_x_axis:
        shared_edges = np.linspace(min(bins[0] for bins in edges.values()),
                                   max(bins[-1] for bins in edges.values()), args.bins + 1)
        edges = {layer: shared_edges.copy() for layer in LAYERS}
    histograms = []
    for path, summary in zip(args.input, summaries):
        print(f"Counting histograms: {path}", flush=True)
        counts = count_histograms(path, edges, args.exclude_special_tokens)
        for layer in LAYERS:
            assert int(counts[layer].sum()) == summary["layers"][layer]["count"], layer
        histograms.append(counts)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_values = histograms
    if args.normalization == "density":
        plot_values = []
        for counts, summary in zip(histograms, summaries):
            densities = {}
            for layer in LAYERS:
                count = summary["layers"][layer]["count"]
                bin_widths = np.diff(edges[layer])
                densities[layer] = counts[layer] / (count * bin_widths)
            plot_values.append(densities)
    shared_y_max = max(values.max() for histogram in plot_values for values in histogram.values())

    def format_axes(ax):
        if args.shared_x_axis:
            ax.set_xlim(min(bins[0] for bins in edges.values()),
                        max(bins[-1] for bins in edges.values()))
        if args.hide_ticks:
            ax.set_xticks([])
        if args.hide_ticks or args.hide_y_ticks:
            ax.set_yticks([])
        if args.shared_y_axis:
            ax.set_ylim(0, shared_y_max * 1.05)

    for layer in LAYERS:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        draw_histogram(ax, layer, edges, plot_values, labels, args.normalization,
                       args.exclude_special_tokens)
        format_axes(ax)
        subtitle = "\nExcluding </s>, <unk>, blank" if args.exclude_special_tokens and layer == "linear2" else ""
        ax.set_title(args.title + subtitle, fontsize=11)
        for extension in ("png", "pdf"):
            fig.savefig(args.output_dir / f"{layer}_histogram.{extension}", dpi=180)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(LAYERS), figsize=(6.5 * len(LAYERS), 4.8),
                             constrained_layout=True, squeeze=False)
    for ax, layer in zip(axes.ravel(), LAYERS):
        draw_histogram(ax, layer, edges, plot_values, labels, args.normalization,
                       args.exclude_special_tokens)
        format_axes(ax)
    fig.suptitle(args.title)
    combined_name = "prediction_histograms.png" if list(LAYERS) == ["predictor_linear1"] else "joiner_histograms.png"
    fig.savefig(args.output_dir / combined_name, dpi=180)
    plt.close(fig)
    report = {
        "labels": labels,
        "normalization": args.normalization,
        "bins": args.bins,
        "hide_ticks": args.hide_ticks,
        "hide_y_ticks": args.hide_y_ticks,
        "shared_x_axis": args.shared_x_axis,
        "shared_y_axis": args.shared_y_axis,
        "excluded_linear2_tokens": ["</s> (0)", "<unk> (1)", "blank (last)"] if args.exclude_special_tokens else [],
        "linear2_measurement": "post-output-quantization" if args.post_output_quantization else "pre-output-quantization",
        "inputs": summaries,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    arrays = {f"{layer}_edges": bins for layer, bins in edges.items()}
    arrays.update({f"input_{i}_{layer}_counts": counts[layer]
                   for i, counts in enumerate(histograms) for layer in LAYERS})
    np.savez_compressed(args.output_dir / "histogram_counts.npz", **arrays)
    print(json.dumps(report, indent=2))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
