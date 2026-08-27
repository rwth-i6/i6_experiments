import re
from typing import List, Optional, Union

from sisyphus import tk, Job, Task

from i6_core.returnn.training import PtCheckpoint


class CalculatePruningStatsJob(Job):
    """
    Loads a PyTorch checkpoint and computes weight pruning / sparsity statistics.

    The memristor pruning setups apply the pruning mask at runtime
    (``weight * (weight.abs() >= threshold/cutoff)``), so the weights in a training
    checkpoint are not zeroed on disk. Pass ``threshold`` or ``percentile`` to
    re-apply the mask condition and count the weights that would be pruned.
    Exact zeros are always counted as well, so the job can also be pointed at an
    already pruned checkpoint (e.g. pruned_model.pt) without threshold/percentile.

    Note: for setups that prune after quantization the runtime mask acts on the
    quantized weights; this job applies it to the raw weights, which is a close
    approximation.
    """

    # weight magnitudes are roughly log-distributed, so log-spaced edges resolve the
    # relevant range; 0.02 is added since it is a threshold actually used in the sweeps
    DEFAULT_HISTOGRAM_BUCKETS = [1e-4, 3e-4, 1e-3, 3e-3, 0.01, 0.02, 0.03, 0.1, 0.3, 1.0]

    def __init__(
        self,
        checkpoint: Union[PtCheckpoint, tk.Path],
        threshold: Optional[float] = None,
        percentile: Optional[float] = None,
        weight_name_patterns: Optional[List[str]] = None,
        histogram_buckets: Optional[List[float]] = None,
        per_layer_histogram: bool = False,
    ):
        """
        :param checkpoint: checkpoint to analyze
        :param threshold: count weights with abs value below this fixed threshold as pruned
        :param percentile: count the bottom percentile fraction (in [0, 1]) per tensor as pruned;
            if both threshold and percentile are given, the union of both masks is counted
            (pruned iff abs value below max(percentile cutoff, threshold))
        :param weight_name_patterns: regex patterns selecting the prunable tensors by name,
            defaults to all tensors named *weight with at least 2 dims (the quantized linears)
        :param histogram_buckets: ascending positive upper bucket edges; they are mirrored to
            the negative side so the histogram is symmetric around a dedicated exact-zero
            bucket, with underflow/overflow buckets beyond the outermost edges
        :param per_layer_histogram: additionally include a histogram per matched tensor
        """
        self.checkpoint = checkpoint
        self.threshold = threshold
        self.percentile = percentile
        self.weight_name_patterns = weight_name_patterns
        self.histogram_buckets = histogram_buckets if histogram_buckets is not None else self.DEFAULT_HISTOGRAM_BUCKETS
        assert list(self.histogram_buckets) == sorted(self.histogram_buckets) and all(
            e > 0 for e in self.histogram_buckets
        ), "histogram_buckets must be ascending positive edges"
        self.per_layer_histogram = per_layer_histogram

        self.out_pruning_percent = self.output_var("pruning_percent")
        self.out_zero_percent = self.output_var("zero_percent")
        self.out_stats = self.output_var("stats")
        self.out_report = self.output_path("pruning_stats.txt")
        self.out_plot = self.output_path("histogram.png")

        self.rqmt = {"cpu": 1, "mem": 8, "time": 1}

    def tasks(self):
        yield Task("run", mini_task=True)

    def _matches(self, name: str, tensor) -> bool:
        if self.weight_name_patterns is not None:
            return any(re.search(pattern, name) for pattern in self.weight_name_patterns)
        return name.endswith("weight") and tensor.dim() >= 2

    def run(self):
        import torch

        state = torch.load(str(self.checkpoint), map_location=torch.device("cpu"))
        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        # symmetric histogram: mirror the positive edges, exact zeros get their own bucket
        edges = [float(e) for e in self.histogram_buckets]
        boundaries = torch.tensor([-e for e in reversed(edges)] + [0.0] + edges)
        n_neg_buckets = len(edges) + 1
        hist_labels = (
            [f"(-inf,{-edges[-1]:g}]"]
            + [f"({-edges[i + 1]:g},{-edges[i]:g}]" for i in reversed(range(len(edges) - 1))]
            + [f"({-edges[0]:g},0)", "0", f"(0,{edges[0]:g}]"]
            + [f"({edges[i]:g},{edges[i + 1]:g}]" for i in range(len(edges) - 1)]
            + [f"({edges[-1]:g},inf)"]
        )

        def _histogram(flat, zeros):
            idx = torch.bucketize(flat[flat != 0], boundaries)
            counts = torch.bincount(idx, minlength=len(boundaries) + 1).tolist()
            return counts[:n_neg_buckets] + [zeros] + counts[n_neg_buckets:]

        hist_counts = [0] * len(hist_labels)
        per_tensor = {}
        total_params = 0
        matched_params = 0
        matched_zeros = 0
        matched_pruned = 0
        for name, tensor in state.items():
            if not torch.is_tensor(tensor) or not tensor.is_floating_point():
                continue
            total_params += tensor.numel()
            if not self._matches(name, tensor):
                continue
            tensor = tensor.float()
            numel = tensor.numel()
            zeros = int((tensor == 0).sum())
            tensor_hist = _histogram(tensor.flatten(), zeros)
            hist_counts = [a + b for a, b in zip(hist_counts, tensor_hist)]
            if self.threshold is not None or self.percentile is not None:
                cutoff = self.threshold if self.threshold is not None else 0.0
                if self.percentile is not None:
                    abs_flat = tensor.abs().flatten()
                    if numel < 2**24:  # torch.quantile is limited to 2**24 elements
                        pctile_cutoff = torch.quantile(abs_flat, self.percentile)
                    else:
                        pctile_cutoff = torch.kthvalue(abs_flat, max(1, int(round(self.percentile * numel)))).values
                    cutoff = max(cutoff, float(pctile_cutoff))
                pruned = int((tensor.abs() < cutoff).sum())
            else:
                pruned = zeros
            matched_params += numel
            matched_zeros += zeros
            matched_pruned += pruned
            per_tensor[name] = {
                "shape": list(tensor.shape),
                "numel": numel,
                "zeros": zeros,
                "pruned": pruned,
                "zero_percent": 100.0 * zeros / numel,
                "pruning_percent": 100.0 * pruned / numel,
            }
            if self.per_layer_histogram:
                per_tensor[name]["histogram"] = dict(zip(hist_labels, tensor_hist))

        assert matched_params > 0, "no tensors matched the weight selection"
        pruning_percent = 100.0 * matched_pruned / matched_params
        zero_percent = 100.0 * matched_zeros / matched_params

        self.out_pruning_percent.set(pruning_percent)
        self.out_zero_percent.set(zero_percent)
        self.out_stats.set(
            {
                "total_params": total_params,
                "matched_params": matched_params,
                "matched_zeros": matched_zeros,
                "matched_pruned": matched_pruned,
                "pruning_percent": pruning_percent,
                "zero_percent": zero_percent,
                "threshold": self.threshold,
                "percentile": self.percentile,
                "histogram": dict(zip(hist_labels, hist_counts)),
                "per_tensor": per_tensor,
            }
        )

        with open(self.out_report.get_path(), "wt") as f:
            f.write(f"checkpoint: {self.checkpoint}\n")
            f.write(f"threshold: {self.threshold}  percentile: {self.percentile}\n")
            f.write(f"total params (all float tensors): {total_params}\n")
            f.write(f"matched params: {matched_params} ({100.0 * matched_params / total_params:.1f}% of total)\n")
            f.write(f"pruned: {matched_pruned} ({pruning_percent:.2f}% of matched)\n")
            f.write(f"exact zeros: {matched_zeros} ({zero_percent:.2f}% of matched)\n\n")
            f.write("weight value histogram (all matched tensors):\n")
            label_len = max(len(label) for label in hist_labels)
            for label, count in zip(hist_labels, hist_counts):
                f.write(f"{label.ljust(label_len)}  {count:>12}  {100.0 * count / matched_params:>7.3f}%\n")
            f.write("\n")
            name_len = max(len(name) for name in per_tensor)
            f.write(f"{'name'.ljust(name_len)}  {'shape'.ljust(20)}  {'numel':>10}  {'zero%':>7}  {'pruned%':>7}\n")
            for name, stats in per_tensor.items():
                f.write(
                    f"{name.ljust(name_len)}  {str(stats['shape']).ljust(20)}  {stats['numel']:>10}"
                    f"  {stats['zero_percent']:>7.2f}  {stats['pruning_percent']:>7.2f}\n"
                )

        self._plot_histogram(hist_labels, hist_counts, pruning_percent)

    def _plot_histogram(self, hist_labels, hist_counts, pruning_percent):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(range(len(hist_labels)), hist_counts, color="#4477aa", width=0.8)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_xticks(range(len(hist_labels)))
        ax.set_xticklabels(hist_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("count (log scale)")
        ax.set_xlabel("weight value bucket")
        title = f"weight value histogram, pruned: {pruning_percent:.2f}%"
        if self.threshold is not None:
            title += f" (threshold {self.threshold:g})"
        elif self.percentile is not None:
            title += f" (percentile {self.percentile:g})"
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fig.savefig(self.out_plot.get_path(), dpi=150)
        plt.close(fig)
