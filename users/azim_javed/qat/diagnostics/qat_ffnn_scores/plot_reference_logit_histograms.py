"""Compare saved pre-softmax logits at shared reference-conditioned states.

Uses the intersection of the two manifests and validates shapes, reference
labels, and blank IDs. Each experiment retains its own encoder/predictor.
Writes both all-token and ordinary-token-only histograms with shared bins.
One pass, exact counts, bounded memory, and resumable checkpoints. The default
range is [-64, 64]; values outside it cause an error rather than being dropped.
"""

import argparse
import json
import hashlib
import os
from time import perf_counter
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = ("all_tokens", "ordinary_tokens")


def iter_pairs(dirs, manifests, segments):
    for tag in segments:
        outputs, references, blanks = [], [], []
        for directory, manifest in zip(dirs, manifests):
            with np.load(directory / manifest[tag]["file"]) as data:
                outputs.append(np.asarray(data["logits"]))
                references.append(np.asarray(data["labels"]))
                blanks.append(int(data["blank_label"]))
            if list(outputs[-1].shape) != manifest[tag]["logits_shape"]:
                raise ValueError(f"Manifest shape mismatch: {tag}")
        if outputs[0].shape != outputs[1].shape or not np.array_equal(references[0], references[1]):
            raise ValueError(f"Unmatched reference states: {tag}")
        if outputs[0].ndim != 3 or blanks[0] != blanks[1] or blanks[0] != outputs[0].shape[-1] - 1:
            raise ValueError(f"Unexpected vocabulary/blank layout: {tag}")
        if outputs[0].shape[1] != len(references[0]) + 1:
            raise ValueError(f"Unexpected reference-position dimension: {tag}")
        if outputs[0].shape[-1] < 4 or any(not np.issubdtype(output.dtype, np.floating) for output in outputs):
            raise ValueError(f"Unexpected logit dimensions/dtype: {tag}")
        yield outputs


def summarize_array(logits, bins, value_range):
    low, high = float(logits.min()), float(logits.max())
    if not np.isfinite([low, high]).all():
        raise ValueError("Nonfinite logits")
    if low < value_range[0] or high > value_range[1]:
        raise ValueError(f"Logit range [{low}, {high}] exceeds --value-range {value_range}")
    # Integer bins + range selects NumPy's uniform-bin/bincount path. Supplying
    # an edge array instead selects a much slower sorting-based algorithm.
    all_counts = np.histogram(logits, bins=bins, range=value_range)[0]
    special = logits[..., [0, 1, -1]]
    special_counts = np.histogram(special, bins=bins, range=value_range)[0]
    total = float(np.sum(logits, dtype=np.float64))
    # Compute the second moment without allocating a full float64 copy or a
    # full-sized squared/centered temporary array.
    squared = float(np.einsum("ijk,ijk->", logits, logits, dtype=np.float64, optimize=False))
    special_total = float(np.sum(special, dtype=np.float64))
    special_squared = float(np.einsum("ijk,ijk->", special, special, dtype=np.float64, optimize=False))
    ordinary = logits[..., 2:-1]
    result = {
        "all_tokens": (all_counts, logits.size, total, squared, low, high),
        "ordinary_tokens": (all_counts - special_counts, ordinary.size,
                            total - special_total, squared - special_squared,
                            float(ordinary.min()), float(ordinary.max())),
    }
    for counts, n, *_ in result.values():
        if int(counts.sum()) != n or np.any(counts < 0):
            raise ValueError("Histogram count mismatch")
    return result


def update_stats(stats, n, value_sum, squared_sum, low, high):
    mean = value_sum / n
    m2 = max(0.0, squared_sum - value_sum * mean)
    total = stats["count"] + n
    delta = mean - stats["mean"]
    stats["m2"] += m2 + delta ** 2 * stats["count"] * n / total
    stats["mean"] += delta * n / total
    stats["count"] = total
    stats["min"] = min(stats["min"], low)
    stats["max"] = max(stats["max"], high)


def save_checkpoint(path, signature, processed, states, stats, counts):
    metadata = {"signature": signature, "processed": processed, "states": states, "stats": stats}
    arrays = {f"input_{i}_{mode}": model_counts[mode]
              for i, model_counts in enumerate(counts) for mode in MODES}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez(stream, metadata=np.asarray(json.dumps(metadata)), **arrays)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--digital-dir", type=Path, required=True)
    parser.add_argument("--memristor-dir", type=Path, required=True)
    parser.add_argument("--bins", type=int, default=128)
    parser.add_argument("--value-range", nargs=2, type=float, default=[-64.0, 64.0], metavar=("MIN", "MAX"),
                        help="Fixed shared boundaries (default: -64 64); rejects out-of-range values")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoint in the output directory")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "reference_logit_histograms")
    args = parser.parse_args()
    if args.bins < 1 or args.checkpoint_every < 1:
        parser.error("Bin count and checkpoint interval must be positive")
    if not np.isfinite(args.value_range).all() or args.value_range[0] >= args.value_range[1]:
        parser.error("Value range must have MIN < MAX")
    dirs = [args.digital_dir, args.memristor_dir]
    manifests = [json.loads((directory / "manifest.json").read_text()) for directory in dirs]
    segments = sorted(set(manifests[0]) & set(manifests[1]))
    if not segments:
        parser.error("No shared utterances")
    print(f"Using {len(segments)} shared utterances from manifests of sizes "
          f"{len(manifests[0])} and {len(manifests[1])}", flush=True)
    stats = [{mode: {"count": 0, "mean": 0.0, "m2": 0.0,
                     "min": float("inf"), "max": -float("inf")} for mode in MODES}
             for _ in dirs]
    edges = {mode: np.linspace(*args.value_range, args.bins + 1) for mode in MODES}
    counts = [{mode: np.zeros(args.bins, dtype=np.int64) for mode in MODES} for _ in dirs]
    states = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "checkpoint.npz"
    signature_data = {"version": 1, "directories": [str(d.resolve()) for d in dirs],
                      "manifests": manifests, "bins": args.bins, "range": args.value_range}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    start = 0
    if args.resume:
        with np.load(checkpoint, allow_pickle=False) as saved:
            metadata = json.loads(saved["metadata"].item())
            if metadata["signature"] != signature:
                parser.error("Checkpoint inputs or bin settings differ from this invocation")
            start, states, stats = metadata["processed"], metadata["states"], metadata["stats"]
            counts = [{mode: saved[f"input_{i}_{mode}"].copy() for mode in MODES} for i in range(2)]
        print(f"Resuming after {start}/{len(segments)} utterances", flush=True)
    elif checkpoint.exists():
        parser.error("Checkpoint exists; use --resume or a new output directory")
    if not 0 <= start <= len(segments):
        parser.error("Invalid checkpoint progress")
    began = perf_counter()
    for processed, outputs in enumerate(iter_pairs(dirs, manifests, segments[start:]), start=start + 1):
        states += outputs[0].shape[0] * outputs[0].shape[1]
        for output, model_stats, model_counts in zip(outputs, stats, counts):
            for mode, (histogram, n, value_sum, squared_sum, low, high) in summarize_array(
                output, args.bins, args.value_range
            ).items():
                update_stats(model_stats[mode], n, value_sum, squared_sum, low, high)
                model_counts[mode] += histogram
        if processed % args.checkpoint_every == 0 or processed == len(segments):
            save_checkpoint(checkpoint, signature, processed, states, stats, counts)
        if processed == start + 1 or processed % args.checkpoint_every == 0 or processed == len(segments):
            elapsed = perf_counter() - began
            eta = (len(segments) - processed) * elapsed / (processed - start)
            print(f"Processed {processed}/{len(segments)}; elapsed {elapsed:.1f}s; ETA {eta / 60:.1f}min", flush=True)

    labels = ["Digital QAT", "Memristor, ADC 4/4"]
    arrays = {}
    for mode in MODES:
        fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
        for i, (model_counts, model_stats, label) in enumerate(zip(counts, stats, labels)):
            s = model_stats[mode]
            assert int(model_counts[mode].sum()) == s["count"]
            s["std"] = float(np.sqrt(s.pop("m2") / s["count"]))
            density = model_counts[mode] / (s["count"] * np.diff(edges[mode]))
            ax.stairs(density, edges[mode], fill=True, alpha=0.75,
                      color=["#447fb8", "#edb637"][i], label=label)
            arrays[f"input_{i}_{mode}_counts"] = model_counts[mode]
        ax.set_xlabel("Joiner output logit (before softmax)")
        ax.set_ylabel("Probability density")
        if args.value_range:
            ax.set_xlim(min(s[mode]["min"] for s in stats), max(s[mode]["max"] for s in stats))
        suffix = "All tokens" if mode == "all_tokens" else "Excluding </s>, <unk>, blank"
        ax.set_title(f"Reference-conditioned joiner outputs · {len(segments)} utterances\n{suffix}", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        for extension in ("png", "pdf"):
            fig.savefig(args.output_dir / f"{mode}_histogram.{extension}", dpi=180)
        plt.close(fig)
        arrays[f"{mode}_edges"] = edges[mode]
    report = {"measurement": "complete joint_net output, before softmax",
              "states_per_model": states, "shared_utterances": len(segments), "segments": segments,
              "bins": args.bins, "value_range": args.value_range,
              "inputs": [{"label": label, "directory": str(directory.resolve()),
                          "manifest_utterances": len(manifest), "statistics": s}
                         for label, directory, manifest, s in zip(labels, dirs, manifests, stats)]}
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    np.savez_compressed(args.output_dir / "histogram_counts.npz", **arrays)
    print(json.dumps({"states_per_model": states, "shared_utterances": len(segments),
                      "statistics": dict(zip(labels, stats))}, indent=2))
    print(f"Saved plots to {args.output_dir}")


if __name__ == "__main__":
    main()
