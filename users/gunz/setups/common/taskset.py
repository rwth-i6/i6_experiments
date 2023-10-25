import os
import textwrap
from typing import Iterator, List, Union

from sisyphus import tk, Job, Task, Path


class WriteTasksetRunScriptJob(Job):
    def __init__(self, binary_path: Path, pin_to_cores: Union[tk.Variable, List[int]]):
        super().__init__()

        self.pin_to_cores = pin_to_cores
        self.binary_path = binary_path

        self.out_script = self.output_path("run-with-task-affinity.sh")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        cores = self.pin_to_cores if isinstance(self.pin_to_cores, list) else self.pin_to_cores.get()
        core_str = ",".join([str(v) for v in cores])

        script = textwrap.dedent(
            f"""
            #!/usr/bin/env bash

            taskset -c {core_str} {self.binary_path} $*
            """
        )

        with open(self.out_script, "wt") as f:
            f.write(script)

        os.chmod(self.out_script, 0o755)
