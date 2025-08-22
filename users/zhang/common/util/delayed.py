from typing import Iterator
from sisyphus import Job, Task
from sisyphus.delayed_ops import Delayed, DelayedBase


class GetWrapper(Delayed):
    def get(self):
        return self.a


class ToVariableJob(Job):
    def __init__(self, v: DelayedBase, pickle: bool = False):
        self.value = v
        self.out = self.output_var("v", pickle=pickle)

    def tasks(self) -> Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        self.out.set(self.value.get())
