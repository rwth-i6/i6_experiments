import json
from typing import Any, Iterator

from sisyphus import Job, Task

from i6_core.util import instanciate_delayed


class DumpAsJsonJob(Job):
    def __init__(self, value: Any):
        self.value = value
        self.out = self.output_path("out.json")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        with open(self.out, "wt") as out_f:
            json.dump(instanciate_delayed(self.value), out_f, indent=2)
