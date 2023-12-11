from typing import List

from sisyphus import tk, Job


class ComputeAverageJob(Job):
    def __init__(self, vals: List[tk.Variable]):
        self.vals = vals

        self.out_avg = self.output_var("avg")

    def run(self):
        vals = [v.get() for v in self.vals]
        self.out_avg.set(sum(vals) / len(vals))
