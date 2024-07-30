import subprocess
import os
import shutil

import numpy as np

from sisyphus import Path, Job, Task

from i6_core.returnn.config import ReturnnConfig
from i6_core import util


class GetSeqLenFileJob(Job):
  def __init__(self, returnn_config: ReturnnConfig, returnn_root: Path, returnn_python_exe: Path, time_rqmt: int = 1):
    self.returnn_config = returnn_config
    self.returnn_root = returnn_root
    self.returnn_python_exe = returnn_python_exe

    self.time_rqmt = time_rqmt

    self.out_seq_len_file = self.output_path("seq_lens")
    self.out_statistics = self.output_path("statistics")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": self.time_rqmt, "gpu": 0})

  def run(self):
    self.returnn_config.write("returnn.config")

    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(self.returnn_root.get_path(), "tools", "dump-dataset.py"),
      "returnn.config",
      "--dataset", "eval",
      "--endseq", "-1",
      "--type", "dump_seq_len",
      "--dump_prefix", "out_"
    ]

    util.create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    seq_len_file = "out_seq-lens.txt"

    with open(seq_len_file, "r") as f:
      with open(self.out_statistics.get_path(), "w+") as stat_file:
        seq_lens = list(eval(f.read()).values())
        assert type(seq_lens) == list

        stat_file.write("Max len: " + str(np.max(seq_lens)) + "\n")
        stat_file.write("Min len: " + str(np.min(seq_lens)) + "\n")
        stat_file.write("Mean len: " + str(np.mean(seq_lens)) + "\n")
        stat_file.write("Std len: " + str(np.std(seq_lens)) + "\n")

    shutil.move(seq_len_file, self.out_seq_len_file.get_path())

  @classmethod
  def hash(cls, kwargs):
    d = {
      "returnn_config": kwargs["returnn_config"],
      "returnn_root": kwargs["returnn_root"],
      "returnn_python_exe": kwargs["returnn_python_exe"],
    }

    return super().hash(d)
