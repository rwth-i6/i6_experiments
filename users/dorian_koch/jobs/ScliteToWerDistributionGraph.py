from typing import List, Tuple, Union
from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import uopen
import math


class ScliteToWerDistributionGraph(Job):

    def __init__(
        self,
        *,
        report_dir: tk.AbstractPath,
        num_bins: int = 10,
        plot_title: Union[str, DelayedBase] = "WER distribution",
        plot_metrics: bool = True,
    ):
        self.report_dir = report_dir
        self.num_bins = num_bins
        self.plot_title = plot_title
        self.plot_metrics = plot_metrics

        self.out_file = self.output_path("vals.csv")
        self.distrib_file = self.output_path("distrib.csv")
        self.out_plot = self.output_path("plot.png")
        self.out_plot_no_ylim = self.output_path("plot_no_ylim.png")
        self.out_plot_ylim_without_first_bin = self.output_path("plot_ylim_without_first_bin.png")
        self.out_plot_ylim10p = self.output_path("plot_ylim10p.png")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib.pyplot as plt
        output_dir = self.report_dir.get_path()

        values = []
        
        print("Reading sclite.pra")
        with open(f"{output_dir}/sclite.pra", "rt", errors="ignore") as f:
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

        with uopen(self.out_file, "wt") as out:
            out.write("corrections,substitutions,deletions,insertions\n")
            for (c,s,d,i) in values:
                out.write(f"{c},{s},{d},{i}\n")
        print(f"Wrote to {self.out_file}")

        bins = [0] * self.num_bins
        avg = 0
        total = 0
        for (c,s,d,i) in values:
            if s+d+c == 0:
                print("Warning: empty sequence")
                continue
            wer = 100.0 * (s + d + i) / (s + d + c)
            bin_idx = min(int(wer / 100.0 * self.num_bins), self.num_bins - 1)
            val = (s + d + c)
            avg += wer * val
            bins[bin_idx] += val
            total += val

        print("WER distribution:")
        with uopen(self.distrib_file, "wt") as out:
            out.write("bin_start,bin_end,count,relative_count_weighed_by_ref_length\n")
            for i, count in enumerate(bins):
                print(f"{i/self.num_bins:.4f}-{(i+1)/self.num_bins:.4f}: {count / total * 100:.3f}%")
                out.write(f"{i/self.num_bins:.4f},{(i+1)/self.num_bins:.4f},{count},{count / total:.6f}\n")


        plt.figure(figsize=(8, 8))
        # show relative count
        plt.bar(range(self.num_bins), [count / total for count in bins], align="edge")

        if self.plot_metrics:
            # plot avg (this should be the wer score as reported by sclite)
            plt.axvline(x=avg / total / 100 * self.num_bins, color="red", label=f"WER: {avg / total:.2f}")

        plt.xlabel("WER")
        plt.ylabel("fraction")
        plt.ylim(0, 1)
        if isinstance(self.plot_title, DelayedBase):
            plt.title(self.plot_title.get())
        else:
            plt.title(self.plot_title)
        plt.xticks(range(0, self.num_bins, max(1, self.num_bins // 10)), [f"{i/self.num_bins:.2f}" for i in range(0, self.num_bins, max(1, self.num_bins // 10))])
        plt.grid(axis="y")
        plt.legend(loc="upper right")
        plt.savefig(self.out_plot)
        plt.autoscale(axis="y")
        plt.savefig(self.out_plot_no_ylim)
        new_ylim = max([count / total for count in bins[1:]]) * 1.1
        plt.ylim(0, new_ylim)
        plt.savefig(self.out_plot_ylim_without_first_bin)
        plt.ylim(0, 0.1)
        plt.savefig(self.out_plot_ylim10p)


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
    ):
        self.report_dirs = report_dirs
        self.num_bins_in_each_direction = num_bins_in_each_direction
        self.plot_title = plot_title
        self.plot_metrics = plot_metrics
        self.x_extents = x_extents
        self.ignore_perfect_seqs = ignore_perfect_seqs
        assert len(self.report_dirs) == 2

        #self.out_file = self.output_path("vals.csv")
        #self.distrib_file = self.output_path("distrib.csv")
        self.out_plot = self.output_path("plot.png")
        #self.out_plot_no_ylim = self.output_path("plot_no_ylim.png")
        #self.out_plot_ylim_without_first_bin = self.output_path("plot_ylim_without_first_bin.png")
        #self.out_plot_ylim10p = self.output_path("plot_ylim10p.png")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import matplotlib.pyplot as plt
        output_dirs = [rd.get_path() for rd in self.report_dirs]

        all_vals = []
        for odir in output_dirs:
            values = {}

            print("Reading sclite.pra")
            cur_seq_id = None
            with open(f"{odir}/sclite.pra", "rt", errors="ignore") as f:
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

        """with uopen(self.out_file, "wt") as out:
            out.write("corrections,substitutions,deletions,insertions\n")
            for (c,s,d,i) in values:
                out.write(f"{c},{s},{d},{i}\n")
        print(f"Wrote to {self.out_file}")"""

        bins = [0] * (self.num_bins_in_each_direction * 2 + 1)
        avg = 0
        total = 0
        for seq_tag in list(all_vals[0].keys()):
            (c,s,d,i) = all_vals[0][seq_tag]
            (c1,s1,d1,i1) = all_vals[1][seq_tag]
            assert s+d+c == s1+d1+c1

            if s+d+c == 0:
                print("Warning: empty sequence")
                continue

            wer0 = 100.0 * (s + d + i) / (s + d + c)
            wer1 = 100.0 * (s1 + d1 + i1) / (s1 + d1 + c1)
            if self.ignore_perfect_seqs and wer0 == 0 and wer1 == 0:
                continue
            wer_diff = wer1 - wer0
            if wer_diff < -self.x_extents or wer_diff > self.x_extents:
                continue

            bin_idx = min(int(wer_diff / self.x_extents * self.num_bins_in_each_direction + self.num_bins_in_each_direction), len(bins) - 1)
            val = (s + d + c)
            avg += wer_diff * val
            bins[bin_idx] += val
            total += val

        """print("WER distribution:")
        with uopen(self.distrib_file, "wt") as out:
            out.write("bin_start,bin_end,count,relative_count_weighed_by_ref_length\n")
            for i, count in enumerate(bins):
                print(f"{i/self.num_bins:.4f}-{(i+1)/self.num_bins:.4f}: {count / total * 100:.3f}%")
                out.write(f"{i/self.num_bins:.4f},{(i+1)/self.num_bins:.4f},{count},{count / total:.6f}\n")
        """

        plt.figure(figsize=(8, 8))
        # show relative count
        eps = 1e-15
        plt.bar(range(-self.num_bins_in_each_direction, self.num_bins_in_each_direction + 1), [count / total + eps for count in bins], align="edge")

        if self.plot_metrics:
            # plot avg (this should be the wer score as reported by sclite)
            # TODO this is probably wrong
            plt.axvline(x=avg / total / self.x_extents * self.num_bins_in_each_direction, color="red", linestyle="--", label=f"WER diff{'(only non-perfect seqs)' if self.ignore_perfect_seqs else ''}: {avg / total:.2f}")

        plt.xlabel("WER difference")
        plt.ylabel("fraction")
        plt.ylim(math.sqrt(eps), 1)
        plt.yscale("log")
        if isinstance(self.plot_title, DelayedBase):
            plt.title(self.plot_title.get())
        else:
            plt.title(self.plot_title)
        x_range = range(-self.num_bins_in_each_direction, self.num_bins_in_each_direction + 1, max(1, 2 * self.num_bins_in_each_direction // 10))
        plt.xticks(x_range, [f"{i/self.num_bins_in_each_direction*self.x_extents:.2f}" for i in x_range])
        plt.grid(axis="y")
        plt.legend(loc="upper right")
        if self.ignore_perfect_seqs:
            # small notice
            plt.text(0.0, 0.9, "Note: seqs with WER=0 have not been counted", fontsize=10)#, transform=plt.gca().transAxes)
        plt.savefig(self.out_plot)
        """plt.autoscale(axis="y")
        plt.savefig(self.out_plot_no_ylim)
        new_ylim = max([count / total for count in bins[1:]]) * 1.1
        plt.ylim(0, new_ylim)
        plt.savefig(self.out_plot_ylim_without_first_bin)
        plt.ylim(0, 0.1)
        plt.savefig(self.out_plot_ylim10p)"""
