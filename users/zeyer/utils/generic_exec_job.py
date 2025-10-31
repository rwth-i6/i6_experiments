from __future__ import annotations
from typing import Union, Sequence, Collection
import os
import subprocess as sp
from sisyphus import Job, Task
from sisyphus.delayed_ops import DelayedBase
from i6_experiments.users.zeyer.sis_tools.instanciate_delayed import instanciate_delayed_copy


class GenericExecJob(Job):
    """
    A job that runs a generic executable (e.g. script) with specified parameters.
    """

    def __init__(
        self,
        args: Union[DelayedBase, Sequence[Union[DelayedBase, str]]],
        *,
        non_hashed_args: Union[DelayedBase, Sequence[Union[DelayedBase, str]]] = (),
        output_files: Collection[str] = (),
        output_dirs: Collection[str] = (),
        output_vars: Collection[str] = (),
    ):
        self.args = args
        self.non_hashed_args = non_hashed_args

        self.output_files = {k: self.output_path(k) for k in output_files}
        self.output_dirs = {k: self.output_path(k, directory=True) for k in output_dirs}
        self.output_vars = {k: self.output_var(k) for k in output_vars}

        self.rqmt = {}
        self.supports_resume = True

    @classmethod
    def hash(cls, kwargs):
        kwargs.pop("non_hashed_args")
        return super().hash(kwargs)

    def tasks(self):
        yield Task("run", resume="run" if self.supports_resume else None, rqmt=self.rqmt)

    def run(self):
        args = instanciate_delayed_copy(self.args)
        non_hashed_args = instanciate_delayed_copy(self.non_hashed_args)
        assert isinstance(args, (list, tuple))
        assert isinstance(non_hashed_args, (list, tuple))
        args = list(args) + list(non_hashed_args)
        print("$ " + " ".join(args))
        sp.run(args, check=True)

        for path in list(self.output_files.values()) + list(self.output_dirs.values()):
            assert os.path.exists(path.get_path())
        for path in self.output_dirs.values():
            assert os.path.isdir(path.get_path())
