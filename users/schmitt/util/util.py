from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil
import os
import re

from typing import Iterator

from recipe.i6_experiments.users.schmitt.experiments.swb.transducer import config as config_mod
tools_dir = os.path.join(os.path.dirname(os.path.abspath(config_mod.__file__)), "tools")


class ModifySeqFileJob(Job):
  def __init__(self, seq_file, seqs_to_skip):
    self.seq_file = seq_file
    self.seqs_to_skip = seqs_to_skip

    self.out_seqs_file = self.output_path("seqs")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1}, mini_task=True)

  def run(self):
    if self.seqs_to_skip is None:
      shutil.copy(self.seq_file, self.out_seqs_file.get_path())
    else:
      seqs_to_skip = self.seqs_to_skip.get()
      with open(self.seq_file.get_path(), "r") as src:
        with open(self.out_seqs_file.get_path(), "w+") as dst:
          for line in src:
            if line.strip() not in seqs_to_skip:
              dst.write(line)


class GetLearningRateFromFileJob(Job):
  def __init__(
          self,
          lr_file_path: Path,
          epoch: int
  ):
    self.lr_file_path = lr_file_path
    self.epoch = epoch

    self.out_last_lr = self.output_var("out_last_lr")

  def tasks(self) -> Iterator[Task]:
    yield Task("run", rqmt={"cpu": 1}, mini_task=True)

  def run(self):
    with open(self.lr_file_path.get_path(), "r") as lr_file:
      file_contents = lr_file.read().strip()

    learning_rates = re.findall("learningRate=(0\.\d+)", file_contents)
    learning_rate = float(learning_rates[self.epoch - 1])

    self.out_last_lr.set(learning_rate)
