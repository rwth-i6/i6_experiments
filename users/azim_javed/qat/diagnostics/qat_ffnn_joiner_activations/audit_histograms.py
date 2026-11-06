"""Independently recount a saved comparison and inspect channel-wise broadening.

Uses all saved outputs, one record at a time. Never pairs independent searches.
The optional digital checkpoint identifies completely zero quantized weight rows.
"""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_histograms import LAYERS, iter_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--digital-checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads((args.comparison / "summary.json").read_text())
    archived = np.load(args.comparison / "histogram_counts.npz")
    report = {"comparison": str(args.comparison.resolve()), "inputs": []}
    thresholds = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 3.0])
    channel_stats = []
    for i, source in enumerate(summary["inputs"]):
        print(f"Recounting {summary['labels'][i]}", flush=True)
        stats = {}
        for layer in LAYERS:
            width, = source["layers"][layer]["widths"]
            stats[layer] = dict(rows=0, sum=np.zeros(width), sumsq=np.zeros(width),
                                near=np.zeros(len(thresholds), dtype=np.int64),
                                counts=np.zeros(summary["bins"], dtype=np.int64),
                                min=float("inf"), max=-float("inf"))
        records = 0
        for record in iter_records(Path(source["path"])):
            records += 1
            assert len(record["linear1"]) == len(record["linear2"])
            for a, b in zip(record["linear1"], record["linear2"]):
                assert a.shape[0] == b.shape[0]
            for layer, s in stats.items():
                if not record[layer]:
                    continue
                x = np.concatenate(record[layer]).astype(np.float64)
                if layer == "linear2" and summary["excluded_linear2_tokens"]:
                    x = x[:, 2:-1]
                assert np.isfinite(x).all()
                s["rows"] += x.shape[0]
                s["sum"] += x.sum(axis=0)
                s["sumsq"] += np.square(x).sum(axis=0)
                s["min"] = min(s["min"], float(x.min()))
                s["max"] = max(s["max"], float(x.max()))
                s["counts"] += np.histogram(x, bins=archived[f"{layer}_edges"])[0]
                absolute = np.abs(x)
                s["near"] += [np.count_nonzero(absolute < t) for t in thresholds]
        item = {"path": source["path"], "records": records, "layers": {}}
        for layer, s in stats.items():
            original = source["layers"][layer]
            n = s["rows"] * len(s["sum"])
            mean = float(s["sum"].sum() / n)
            std = float(np.sqrt(s["sumsq"].sum() / n - mean ** 2))
            assert s["rows"] == original["rows"]
            assert np.isclose(mean, original["mean"], atol=1e-10)
            assert np.isclose(std, original["std"], atol=1e-10)
            assert s["min"] == original["min"] and s["max"] == original["max"]
            assert np.array_equal(s["counts"], archived[f"input_{i}_{layer}_counts"])
            channel_mean = s["sum"] / s["rows"]
            channel_std = np.sqrt(np.maximum(0, s["sumsq"] / s["rows"] - channel_mean ** 2))
            s["channel_mean"], s["channel_std"] = channel_mean, channel_std
            item["layers"][layer] = {
                "all_archived_counts_and_moments_verified": True,
                "rows": s["rows"], "mean": mean, "std": std,
                "fraction_abs_below": dict(zip(map(str, thresholds), (s["near"] / n).tolist())),
                "channel_std_quantiles": dict(zip(["min", "p10", "median", "p90", "max"],
                                                   np.quantile(channel_std, [0, .1, .5, .9, 1]).tolist())),
                "channels_std_below_1e-6": int(np.count_nonzero(channel_std < 1e-6)),
                "mean_within_channel_variance": float(np.mean(channel_std ** 2)),
                "variance_of_channel_means": float(np.var(channel_mean)),
            }
            if layer == "linear2" and not summary["excluded_linear2_tokens"]:
                ordinary_mean = float(channel_mean[2:-1].mean())
                ordinary_var = float(np.mean(channel_std[2:-1] ** 2) + np.var(channel_mean[2:-1]))
                item["layers"][layer]["ordinary_tokens"] = {
                    "mean": ordinary_mean, "std": float(np.sqrt(ordinary_var)),
                    "mean_within_channel_variance": float(np.mean(channel_std[2:-1] ** 2)),
                    "variance_of_channel_means": float(np.var(channel_mean[2:-1])),
                }
                item["layers"][layer]["special_tokens"] = {
                    name: {"mean": float(channel_mean[j]), "std": float(channel_std[j])}
                    for name, j in (("</s>", 0), ("<unk>", 1), ("blank", -1))
                }
        report["inputs"].append(item)
        channel_stats.append(stats)

    if args.digital_checkpoint:
        import torch

        print("Checking digital quantized weight rows", flush=True)
        torch.set_num_threads(2)
        state = torch.load(args.digital_checkpoint, map_location="cpu", weights_only=False, mmap=True)["model"]
        report["digital_weights"] = {}
        for layer, index in (("linear1", 2), ("linear2", 7)):
            prefix = f"joint_net.{index}."
            obs = prefix + "weight_quantizer.observer."
            bound = torch.maximum(state[obs + "min_val"].abs(), state[obs + "max_val"].abs())
            scale = torch.maximum(bound / 7, state[obs + "eps"])
            w = torch.fake_quantize_per_tensor_affine(state[prefix + "weight"], float(scale), 0, -7, 7)
            zero_rows = (w == 0).all(dim=1).numpy()
            details = {"scale": float(scale), "nonzero_fraction": float((w != 0).float().mean()),
                       "zero_weight_rows": int(zero_rows.sum()), "total_rows": len(zero_rows)}
            if zero_rows.any():
                details["zero_weight_rows_output_rms"] = [
                    float(np.sqrt(np.mean(s[layer]["sumsq"][zero_rows] / s[layer]["rows"])))
                    for s in channel_stats
                ]
            report["digital_weights"][layer] = details

    report["channel_comparison"] = {
        layer: {
            "fraction_channels_with_higher_memristor_std": float(np.mean(
                channel_stats[1][layer]["channel_std"] > channel_stats[0][layer]["channel_std"])),
            "median_memristor_to_digital_std_ratio": float(np.median(
                channel_stats[1][layer]["channel_std"] / channel_stats[0][layer]["channel_std"])),
        }
        for layer in LAYERS
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    for ax, layer in zip(axes, LAYERS):
        edges = archived[f"{layer}_edges"]
        for i, label in enumerate(summary["labels"]):
            counts = archived[f"input_{i}_{layer}_counts"]
            ax.stairs(counts / counts.sum() / np.diff(edges), edges, fill=True, alpha=.55,
                      label=label, color=["#447fb8", "#edb637"][i])
        ax.set_xlabel(LAYERS[layer])
        ax.set_ylabel("Probability density")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
    fig.suptitle("W4/A8 dev-other: independent search outputs (not matched inputs)")
    fig.savefig(args.output_dir / "joiner_histograms_density.png", dpi=180)
    plt.close(fig)
    (args.output_dir / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
