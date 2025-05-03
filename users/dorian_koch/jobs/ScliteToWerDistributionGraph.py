from typing import Dict, List, Literal, Optional, Tuple, Union
from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import uopen
import math


def make_bins(
    values, num_bins: int, val_range: Tuple[float, float], bin_align: Union[Literal["center"], Literal["edge"]]
):
    """
    :param bin_align:
    """
    bins = [0] * num_bins
    avg = 0
    total = 0
    num_offscreen = 0
    for value, weight in values:
        if value > val_range[1] or value < val_range[0]:
            num_offscreen += 1
            continue
        if bin_align == "edge":
            bin_idx = min(int((value - val_range[0]) * num_bins / (val_range[1] - val_range[0])), num_bins - 1)
        elif bin_align == "center":
            bin_idx = int((value - val_range[0]) * (num_bins - 1) / (val_range[1] - val_range[0]) + 0.5)
        assert (
            0 <= bin_idx < num_bins
        ), f"bin_idx {bin_idx} out of range {val_range[0]}-{val_range[1]} for value {value}"

        avg += value * weight
        bins[bin_idx] += weight
        total += weight

    return bins, avg, total, num_offscreen


class ScliteToWerDistributionGraph(Job):
    def __init__(
        self,
        *,
        report_dir: Dict[str, tk.AbstractPath],
        num_bins: int = 10,
        plot_title: Union[str, DelayedBase] = "",
        plot_metrics: bool = True,
        kl_divergence: bool = False,
        logscale: bool = True,
        xlim: Optional[Tuple[float, float]] = (0.0, 100.0),
    ):
        assert isinstance(report_dir, dict)

        self.report_dirs = report_dir
        self.num_bins = num_bins
        self.plot_title = plot_title
        self.plot_metrics = plot_metrics
        self.kl_divergence = kl_divergence
        self.log_scale = logscale
        self.xlim = xlim or (0.0, 100.0)

        assert (
            100 % num_bins == 0
        ), "num_bins must be a divisor of 100"  # otherwise some bins will get more values than others

        # self.out_file = self.output_path("vals.csv")
        # self.distrib_file = self.output_path("distrib.csv")
        self.out_plot = self.output_path("plot.pdf")
        self.out_plot_no_ylim = self.output_path("plot_no_ylim.pdf")
        self.out_plot_ylim_without_first_bin = self.output_path("plot_ylim_without_first_bin.pdf")
        self.out_plot_ylim10p = self.output_path("plot_ylim10p.pdf")

        self.out_plot_len = self.output_path("plot_len.pdf")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 21
        return super().hash(d)

    def tasks(self):
        yield Task("run", mini_task=True)

    def read_reportdirs(self, metric: Union[Literal["WER"], Literal["len"]]) -> dict[str, list]:
        name_with_vals = {}
        for name, report_dir in self.report_dirs.items():
            output_dir = report_dir.get_path()

            values = []

            print("Reading sclite.pra")
            with open(f"{output_dir}/sclite.pra", errors="ignore") as f:
                for line in f:
                    if not line.startswith("Scores:"):
                        continue
                    # Scores: (#C #S #D #I) 79 2 2 0
                    parts = line.split()
                    assert len(parts) == 9
                    numbers = [int(p) for p in parts[-4:]]
                    values.append(numbers)
            assert len(values) > 0
            print(f"Read {len(values)} lines")

            binvals = []
            for c, s, d, i in values:
                if s + d + c == 0:
                    print("Warning: empty sequence")
                    continue
                if metric == "WER":
                    value = 100.0 * (s + d + i) / (s + d + c)
                    weight = s + d + c
                elif metric == "len":
                    value = s + c + i  # hyp len
                    weight = 1
                binvals.append((value, weight))
            name_with_vals[name] = binvals
        return name_with_vals

    def make_kl_div_text(self, name_with_vals: dict[str, list], val_range: tuple[int, int]):
        prob_dists = []
        wer_keys = list(name_with_vals.keys())
        for name in wer_keys:
            binvals = name_with_vals[
                name
            ]  # Here we hardcode the num bins to 100 TODO: just do the kl divergence fully instead of this approx
            bins, avg, total, num_offscreen = make_bins(binvals, 100, val_range, bin_align="edge")

            prob_dists.append([count / total for count in bins])
        assert len(prob_dists) >= 2
        assert all([len(prob_dists[0]) == len(prob_dists[i]) for i in range(len(prob_dists))])
        kl_text = ""
        for k_from in range(len(prob_dists) - 1):
            for k_to in range(k_from + 1, len(prob_dists)):
                kl_div = 0.0
                for i in range(len(prob_dists[0])):
                    p = prob_dists[k_from][i]
                    q = prob_dists[k_to][i]
                    if p > 0 and q > 0:
                        kl_div += p * math.log(p / q)
                print(f"KL divergence ({wer_keys[k_from]} -> {wer_keys[k_to]}): {kl_div:.9f}")
                kl_text += f"KL divergence ({wer_keys[k_from]} -> {wer_keys[k_to]}): {kl_div:.5f}\n"
        return kl_text

    def run(self):
        import matplotlib.pyplot as plt

        for metric in ["WER", "len"]:
            metric_range = {"WER": (0, 100), "len": (0, 100)}.get(metric)
            assert metric_range is not None
            name_with_vals = self.read_reportdirs(metric=metric)

            fig, ax = plt.subplots(figsize=(8, 8))

            colors = ["blue", "green", "orange", "brown", "pink", "gray", "purple", "red"]
            BAR_WIDTH = 0.9 / len(name_with_vals)
            offscreen_vals = []
            for i, name in enumerate(name_with_vals.keys()):
                binvals = name_with_vals[name]
                bins, avg, total, num_offscreen = make_bins(binvals, self.num_bins, metric_range, bin_align="edge")
                assert (
                    abs(sum([count / total for count in bins]) - 1.0) < 1e-8
                ), "bins should make a probability distribution"
                xs = range(self.num_bins)
                xs = [float(x) + i * BAR_WIDTH for x in xs]
                ax.bar(
                    xs,
                    [count / total for count in bins],
                    align="edge",
                    width=BAR_WIDTH,
                    label=name,
                    color=colors[i % len(colors)],
                )

                if self.plot_metrics:
                    # plot avg (this should be the wer score as reported by sclite)
                    ax.axvline(
                        x=avg / total / 100 * self.num_bins,
                        color=colors[i % len(colors)],
                        linestyle="--",
                        label=f"{metric} {name}: {avg / total:.2f}",
                    )
                    offscreen_vals.append(num_offscreen)
            if any([n_off > 0 for n_off in offscreen_vals]):
                ax.text(0.0, 0.8, f"Offscreen: {", ".join([str(x) for x in offscreen_vals])} instances", fontsize=10)
            if self.kl_divergence:
                # compute kl divergence
                kl_text = self.make_kl_div_text(name_with_vals, metric_range)

                fig.text(
                    0.5,
                    0.002,
                    kl_text,
                    fontsize=10,
                    ha="center",
                    va="bottom",
                )
                fig.subplots_adjust(bottom=(0.02 * len(kl_text.split("\n")) + 0.05))

            ax.set_xlabel(metric)
            ax.set_ylabel("fraction")

            lower_lim = 0
            if self.log_scale:
                ax.set_yscale("symlog", linthresh=1e-3)
                lower_lim = 1e-3
            ax.set_ylim(lower_lim, 1)

            if isinstance(self.plot_title, DelayedBase):
                ax.set_title(f"{metric} distribution\n" + self.plot_title.get())
            else:
                ax.set_title(f"{metric} distribution\n" + self.plot_title)
            ax.set_xticks(
                range(0, self.num_bins, max(1, self.num_bins // 10)),
                [f"{100 * i/self.num_bins:.2f}" for i in range(0, self.num_bins, max(1, self.num_bins // 10))],
            )
            ax.set_xlim(self.xlim[0] / 100 * self.num_bins, self.xlim[1] / 100 * self.num_bins)
            ax.grid(axis="y")
            ax.legend(loc="upper right", bbox_to_anchor=(0.95, 1))

            if metric == "WER":
                plt.savefig(self.out_plot, bbox_inches="tight")
                ax.autoscale(axis="y")
                plt.savefig(self.out_plot_no_ylim, bbox_inches="tight")
                # new_ylim = max([count / total for count in bins[1:]]) * 1.1
                # plt.ylim(lower_lim, new_ylim)
                # plt.savefig(self.out_plot_ylim_without_first_bin)
                ax.set_ylim(lower_lim, 0.1)
                plt.savefig(self.out_plot_ylim10p, bbox_inches="tight")
            elif metric == "len":
                plt.savefig(self.out_plot_len, bbox_inches="tight")


class CompareTwoScliteWerDistributions(Job):
    """
    Two sclite reports as input, and creates a plot that shows how the WERs of sequences change.
    Both sclite reports need to have the same sequences
    """

    def __init__(
        self,
        *,
        report_dirs: List[tk.AbstractPath],
        num_bins_in_each_direction: int = 5,
        plot_title: Union[str, DelayedBase] = "WER change distribution",
        plot_metrics: bool = True,
        x_extents: float = 25.0,
        ignore_perfect_seqs: bool = False,
        logscale: bool = True,
    ):
        self.report_dirs = report_dirs
        self.num_bins_in_each_direction = num_bins_in_each_direction
        self.plot_title = plot_title
        self.plot_metrics = plot_metrics
        self.x_extents = x_extents
        self.ignore_perfect_seqs = ignore_perfect_seqs
        self.log_scale = logscale
        assert len(self.report_dirs) == 2

        self.out_plot = self.output_path("plot.pdf")
        self.out_ratio_plot = self.output_path("plot_ratio.pdf")
        self.out_plot_scatter = self.output_path("plot_scatter.pdf")
        self.out_plot_scatter2 = self.output_path("plot_scatter2.pdf")
        self.out_plot_scatter3 = self.output_path("plot_scatter3.pdf")
        self.out_plot_scatter3_gauss = self.output_path("plot_scatter3_gauss.pdf")

    @classmethod
    def hash(cls, parsed_args):
        d = dict(**parsed_args)
        d["__version"] = 24
        return super().hash(d)

    def tasks(self):
        yield Task("run", mini_task=True)

    def make_plot(self, values, title, out_path, extents, is_ratio=False):
        import matplotlib.pyplot as plt

        bins, avg, total, num_offscreen = make_bins(
            values, self.num_bins_in_each_direction * 2 + 1, (-extents, extents), bin_align="center"
        )

        plt.figure(figsize=(8, 8))
        # show relative count
        eps = 1e-15
        plt.bar(
            range(-self.num_bins_in_each_direction, self.num_bins_in_each_direction + 1),
            [count / total + eps for count in bins],
            align="center",
        )

        if self.plot_metrics:
            # plot avg (this should be the wer score as reported by sclite)
            plt.axvline(
                x=avg / total / extents * self.num_bins_in_each_direction,
                color="red",
                linestyle="--",
                label=f"avg: {avg / total + (1.0 if is_ratio else 0):.2f}",
            )
            if num_offscreen > 0:
                plt.text(0.0, 0.8, f"Offscreen: {num_offscreen} instances", fontsize=10)

        # plt.xlabel("WER difference")
        plt.ylabel("fraction")
        smallest_possible_val = 0.9 / total if total > 0 else eps
        plt.ylim(smallest_possible_val, 1)

        if self.log_scale:
            plt.yscale("log")
            plt.ylim(1e-3, 1)
        plt.title(self.plot_title + " " + title)
        x_range = range(
            -self.num_bins_in_each_direction,
            self.num_bins_in_each_direction + 1,
            max(1, 2 * self.num_bins_in_each_direction // 10),
        )
        if is_ratio:
            plt.xticks(x_range, [f"{1.0 + i/self.num_bins_in_each_direction*extents:.2f}" for i in x_range])
        else:
            plt.xticks(x_range, [f"{i/self.num_bins_in_each_direction*extents:.2f}" for i in x_range])
        plt.grid(axis="y")
        plt.legend(loc="upper right")
        if self.ignore_perfect_seqs:
            # small notice
            plt.text(
                0.0, 0.9, "Note: seqs with WER=0 have not been counted", fontsize=10
            )  # , transform=plt.gca().transAxes)
        plt.savefig(out_path)

    def read_report_dirs(self) -> list[dict[str, list]]:
        output_dirs = [rd.get_path() for rd in self.report_dirs]

        all_vals = []
        for odir in output_dirs:
            values = {}

            print("Reading sclite.pra")
            cur_seq_id = None
            with open(f"{odir}/sclite.pra", errors="ignore") as f:
                for line in f:
                    if line.startswith("id:"):
                        # id: (dev-clean/422-122949-0034/422-122949-0034-000)
                        cur_seq_id = line.split()[1][1:-1]
                        assert len(cur_seq_id) > 0
                        assert cur_seq_id not in values
                    elif line.startswith("Scores:"):
                        assert cur_seq_id is not None
                        # Scores: (#C #S #D #I) 79 2 2 0
                        parts = line.split()
                        assert len(parts) == 9
                        numbers = [int(p) for p in parts[-4:]]
                        values[cur_seq_id] = numbers
                        cur_seq_id = None
            assert len(values) > 0
            print(f"Read {len(values)} lines")
            all_vals.append(values)

        assert set(all_vals[0].keys()) == set(all_vals[1].keys()), "seq tags do not match"
        return all_vals

    def run(self):
        if isinstance(self.plot_title, DelayedBase):
            self.plot_title = self.plot_title.get()
        import matplotlib.pyplot as plt
        import matplotlib
        import matplotlib.colors as mcolors
        import os

        # remove current plots
        if os.path.exists(self.out_plot):
            os.remove(self.out_plot)
        if os.path.exists(self.out_ratio_plot):
            os.remove(self.out_ratio_plot)

        all_vals = self.read_report_dirs()

        values = []  # (weight, (wer0, wer1))
        num_wers_ignored = 0
        for seq_tag in list(all_vals[0].keys()):
            (c, s, d, i) = all_vals[0][seq_tag]
            (c1, s1, d1, i1) = all_vals[1][seq_tag]
            assert s + d + c == s1 + d1 + c1

            if s + d + c == 0:
                print("Warning: empty sequence")
                continue

            wer0 = 100.0 * (s + d + i) / (s + d + c)
            wer1 = 100.0 * (s1 + d1 + i1) / (s1 + d1 + c1)
            if self.ignore_perfect_seqs and wer0 == 0 and wer1 == 0:
                num_wers_ignored += 1
                continue
            values.append((s + d + c, (wer0, wer1)))

            # print(f"seq {seq_tag}: {wer0:.2f} -> {wer1:.2f} ({wer_diff:.2f}, {wer_ratio:.2f})")
            print(f"seq {seq_tag}: {wer0:.2f} -> {wer1:.2f}")

        if self.ignore_perfect_seqs:
            print(
                f"Ignored {num_wers_ignored} (of {len(all_vals[0].keys())}) because they were perfect and ignore_perfect_seqs=True"
            )

        values_diff = []  # (value, weight)
        values_ratio = []
        for weight, (wer0, wer1) in values:
            wer_diff = wer1 - wer0

            values_diff.append((wer_diff, weight))

            if abs(wer0) > 1e-8:
                wer_ratio = wer1 / wer0 - 1.0  # do -1.0 so we can just reuse the code for the wer diff
                values_ratio.append((wer_ratio, weight))
            elif wer0 == 0 and wer1 == 0:
                wer_ratio = 0.0
            else:
                wer_ratio = math.nan

        self.make_plot(values_diff, f"WER differences +-{self.x_extents}", self.out_plot, self.x_extents)
        self.make_plot(
            values_ratio, f"relative WER difference", self.out_ratio_plot, self.x_extents / 100, is_ratio=True
        )

        import numpy as np

        # Unpack the data
        weights, coords = zip(*values)
        x, y = zip(*coords)

        # Convert to numpy arrays
        x = np.array(x)
        y = np.array(y)
        weights = np.array(weights)

        # plt reset everything
        LIM = 30
        plt.figure(figsize=(8, 8))

        plt.hexbin(
            x,
            y,
            C=weights,
            reduce_C_function=np.sum,
            gridsize=50,
            cmap="plasma",
            norm=mcolors.LogNorm(vmin=1),
        )
        plt.colorbar(label="Sum of seq lengths")

        plt.xlabel("from")
        plt.ylabel("to")
        plt.xlim(0, LIM)
        plt.ylim(0, LIM)
        plt.title(self.plot_title)
        plt.savefig(self.out_plot_scatter)

        # plt reset everything
        plt.figure(figsize=(8, 8))

        plt.hist2d(x, y, bins=100, weights=weights, cmap="plasma", norm=mcolors.LogNorm(vmin=1))
        plt.colorbar(label="Sum of seq lengths")
        plt.xlabel("from")
        plt.ylabel("to")
        plt.xlim(0, LIM)
        plt.ylim(0, LIM)
        plt.title(self.plot_title)
        plt.savefig(self.out_plot_scatter2)

        plt.figure(figsize=(8, 8))

        plt.scatter(x, y, s=10, alpha=0.8, c=weights, cmap="plasma", norm=mcolors.LogNorm(vmin=1, vmax=100))
        plt.colorbar(label="Seq length (log scale)")
        plt.xlabel("from")
        plt.ylabel("to")
        plt.title(self.plot_title)
        plt.xlim(0, LIM)
        plt.ylim(0, LIM)
        plt.grid()
        plt.savefig(self.out_plot_scatter3)

        plt.figure(figsize=(8, 8))

        # add some gauss noise to x and y
        x_noised = x + np.random.normal(0, 0.3, len(x))
        y_noised = y + np.random.normal(0, 0.3, len(y))
        plt.scatter(
            x_noised, y_noised, s=5, alpha=0.5, c=weights, cmap="plasma", norm=mcolors.LogNorm(vmin=1, vmax=100)
        )
        plt.colorbar(label="Seq length (log scale) with some gauss noise")
        plt.xlabel("from")
        plt.ylabel("to")
        plt.title(self.plot_title)
        plt.xlim(0, LIM)
        plt.ylim(0, LIM)
        plt.grid()
        plt.savefig(self.out_plot_scatter3_gauss)
