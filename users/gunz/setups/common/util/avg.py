from typing import Iterator, List

from sisyphus import tk, Job, Task


class ComputeAverageJob(Job):
    def __init__(self, vals: List[tk.Variable]):
        self.vals = vals

        self.out_avg = self.output_var("avg")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        vals = [v.get() for v in self.vals]
        self.out_avg.set(sum(vals) / len(vals))
