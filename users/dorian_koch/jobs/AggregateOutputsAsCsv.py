from typing import List, Tuple
from sisyphus import Job, Task, tk
from i6_core.util import uopen


class AggregateOutputsAsCsv(Job):
    def __init__(
        self,
        *,
        inputs: list[tuple[str, tk.AbstractPath]],
    ):
        self.inputs = inputs

        self.out_file = self.output_path("out.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        data = []

        for key, file in self.inputs:
            val = None
            if isinstance(file, tk.Variable):
                val = str(file.get())
            elif isinstance(file, tk.Path):
                with uopen(file, "rt") as f:
                    val = f.read()
            else:
                assert False
            data.append((key, val.strip()))

        with uopen(self.out_file, "wt") as out:
            out.write("key,value\n")
            for val, content in data:
                out.write(f"{val},{content}\n")


class WriteFinishedPathsAsCsv(Job):
    def __init__(
        self,
        *,
        inputs: list[tuple[str, tk.AbstractPath | str]],
        seperator: str = ",",
    ):
        self.inputs = inputs
        self.seperator = seperator
        self.out_file = self.output_path("out.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        data = []

        for key, file in self.inputs:
            val = None
            if isinstance(file, tk.Variable):
                val = str(file.get())
            elif isinstance(file, tk.Path):
                val = str(file)
            else:
                assert False
            data.append((key, val.strip()))

        with uopen(self.out_file, "wt") as out:
            out.write(f"key{self.seperator}value\n")
            for val, content in data:
                out.write(f"{val}{self.seperator}{content}\n")
