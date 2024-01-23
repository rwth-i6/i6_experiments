"""GroupJob"""

from __future__ import annotations
from sisyphus import Job, Task


class GroupJob(Job):
    """
    Like tf.group. Depends on a number of inputs.
    It is itself a no-op.
    Because Sisyphus needs an output, it creates a dummy output.
    """

    def __init__(self, inputs):
        super(GroupJob, self).__init__()
        self.inputs = inputs
        self.output = self.output_path("dummy.txt")

    def tasks(self):
        """tasks"""
        yield Task("run", resume="run", mini_task=True)

    def run(self):
        """run"""
        with open(self.output.get_path(), "wt"):
            pass
