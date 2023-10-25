import os
import textwrap
from typing import Iterator, List, Union

from sisyphus import tk, Job, Task, Path


class WriteTasksetRunScriptJob(Job):
    def __init__(
        self,
        rasr_binary_path: Path,
        cores: Union[tk.Variable, List[int]],
    ):
        super().__init__()

        self.cores = cores
        self.rasr_binary_path = rasr_binary_path

        self.out_script = self.output_path("run-with-task-affinity.sh")

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        core_str = ",".join(self.cores if isinstance(self.cores, list) else self.cores.get())

        script = textwrap.dedent(
            f"""
            #!/usr/bin/env bash

            taskset -c {core_str} {self.rasr_binary_path} $*
            """
        )

        with open(self.out_script, "wt") as f:
            f.write(script)

        os.chmod(self.out_script, 0o755)
