from typing import List, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen


class AggregateOutputsAsCsv(Job):

    def __init__(
        self,
        *,
        inputs: List[Tuple[str, tk.Path]],
    ):
        self.inputs = inputs

        self.out_file = self.output_path("out.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        """run"""
        import sys

        data = []

        for val, file in self.inputs:
            with uopen(file, "rt") as f:
                data.append((val, f.read().strip()))

        with uopen(self.out_file, "wt") as out:
            for val, content in data:
                out.write(f"{val},{content}\n")
