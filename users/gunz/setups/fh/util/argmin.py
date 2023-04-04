import typing

from sisyphus import Job, Task


K = typing.TypeVar("K")
V = typing.TypeVar("V")


class ComputeArgminJob(Job, typing.Generic[K, V]):
    def __init__(self, dictionary: typing.Dict[K, V]):
        self.dict = dictionary

        self.out_min = self.output_var("min", pickle=False)
        self.out_argmin = self.output_var("argmin", pickle=False)

        self.rqmt = None

    def tasks(self) -> typing.Iterator[Task]:
        yield Task("run", mini_task=True)

    def run(self):
        argmin = min(self.dict, key=self.dict.get)

        self.out_argmin.set(argmin)
        self.out_min.set(self.dict[argmin])
