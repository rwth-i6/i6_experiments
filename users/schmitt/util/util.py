from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil
import os

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