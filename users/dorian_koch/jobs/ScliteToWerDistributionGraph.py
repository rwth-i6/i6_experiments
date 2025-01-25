from typing import List, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen


class ScliteToWerDistributionGraph(Job):

    def __init__(
        self,
        *,
        report_dir: tk.AbstractPath,
        num_bins: int = 10,
    ):
        self.report_dir = report_dir
        self.num_bins = num_bins

        self.out_file = self.output_path("vals.csv")
        self.distrib_file = self.output_path("distrib.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
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

        for (c,s,d,i) in values:
            if s+d+c == 0:
                print("Warning: empty sequence")
                continue
            wer = 100.0 * (s + d + i) / (s + d + c)
            bin_idx = min(int(wer / 100.0 * self.num_bins), self.num_bins - 1)
            bins[bin_idx] += 1

        print("WER distribution:")
        for i, count in enumerate(bins):
            print(f"{i/self.num_bins:.2f}-{(i+1)/self.num_bins:.2f}: {count / len(values) * 100:.2f}%")
        
