from typing import List, Tuple, Union
from sisyphus import Job, Task, tk
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import uopen


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

