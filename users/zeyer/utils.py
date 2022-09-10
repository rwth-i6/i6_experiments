"""generic utils"""

from __future__ import annotations
from typing import Union, TextIO, BinaryIO
from sisyphus import Job, Task


def generic_open(filename: str, mode: str = "r") -> Union[TextIO, BinaryIO]:
  """
  Wrapper around :func:`open`.
  Automatically wraps :func:`gzip.open` if filename ends with ``".gz"``.

  :param str filename:
  :param str mode: text mode by default
  :rtype: typing.TextIO|typing.BinaryIO
  """
  if filename.endswith(".gz"):
    import gzip
    if "b" not in mode:
      mode += "t"
    return gzip.open(filename, mode)
  return open(filename, mode)


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
