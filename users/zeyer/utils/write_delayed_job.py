"""
Write some delayed object to a tk.Variable
"""


from typing import Union, Any
from sisyphus import Job, Task
from sisyphus.delayed_ops import DelayedBase
from i6_core.util import instanciate_delayed


class DelayedToVariableJob(Job):
    """
    Calculates the delayed object (calls ``.get()``) and writes it to a tk.Variable.
    """

    def __init__(self, delayed: Union[DelayedBase, Any], *, out_filename: str = "out"):
        self.delayed = delayed
        self.out = self.output_var(out_filename)

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

    def run(self):
        val = instanciate_delayed(self.delayed)
        self.out.set(val)
