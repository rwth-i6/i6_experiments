import subprocess
import os
import shutil

import numpy as np

from sisyphus import Path, Job, Task

from i6_core.returnn.config import ReturnnConfig
from i6_core import util


class GetSeqLenFileJob(Job):
  def __init__(
          self,
          returnn_config: ReturnnConfig,
          returnn_root: Path,
          returnn_python_exe: Path,
          time_rqmt: int = 1,
          downsampling_factor: float = 1
  ):
    self.returnn_config = returnn_config
    self.returnn_root = returnn_root
    self.returnn_python_exe = returnn_python_exe
    self.downsampling_factor = downsampling_factor

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

        stat_file.write("Statistics (in seconds)\n")
        stat_file.write("Max len: " + str(np.max(seq_lens) * self.downsampling_factor) + "\n")
        stat_file.write("Min len: " + str(np.min(seq_lens) * self.downsampling_factor) + "\n")
        stat_file.write("Mean len: " + str(np.mean(seq_lens) * self.downsampling_factor) + "\n")
        stat_file.write("Std len: " + str(np.std(seq_lens) * self.downsampling_factor) + "\n")
        stat_file.write("Sum len: " + str(np.sum(seq_lens) * self.downsampling_factor) + "\n")
        stat_file.write("Number of sequences: " + str(len(seq_lens)) + "\n")

    shutil.move(seq_len_file, self.out_seq_len_file.get_path())

  @classmethod
  def hash(cls, kwargs):
    d = {
      "returnn_config": kwargs["returnn_config"],
      "returnn_root": kwargs["returnn_root"],
      "returnn_python_exe": kwargs["returnn_python_exe"],
    }

    return super().hash(d)


class GetCorrectDataFilteringJob(Job):
  def __init__(
          self,
          seq_len_file1: Path,
          seq_len_file2: Path,
          max_seq_len1: int,
  ):
    self.seq_len_file1 = seq_len_file1
    self.seq_len_file2 = seq_len_file2
    self.max_seq_len1 = max_seq_len1

    self.out_threshold = self.output_path("threshold")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    seq_lens = {1: [], 2: []}
    for i, seq_len_file in enumerate([self.seq_len_file1, self.seq_len_file2]):
      print(f"Reading sequence lengths {i + 1}")
      with open(seq_len_file.get_path(), "r") as f:
        for n, line in enumerate(f):
          if n % 1_000_000 == 0:
            print(f"Reading line {n}")
          if line.strip() in ["{", "}"]:
            continue
          _, seq_len = line.strip().split(":")
          seq_len = seq_len.split(",")[0]
          seq_lens[i + 1].append(int(seq_len))

    seq_lens1 = seq_lens[1]
    seq_lens2 = seq_lens[2]
    # filter first list
    print("Filtering first list")
    filtered_seq_lens1 = [l for l in seq_lens1 if l <= self.max_seq_len1]
    print(f"Original number of sequences: {len(seq_lens1)}")
    print(f"Number of filtered sequences 1: {len(filtered_seq_lens1)}")
    num1 = len(seq_lens1)
    num_filtered1 = len(filtered_seq_lens1)
    prop1 = num_filtered1 / num1

    # apply same proportion to second list
    num2 = len(seq_lens2)
    num_filtered2 = int(num2 * prop1)

    # sort second list and get threshold length to achieve same amount of filtered data
    print("Filtering second list")
    seq_lens2_sorted = sorted(seq_lens2)
    threshold_length = seq_lens2_sorted[num_filtered2]
    filtered_seq_lens2 = [l for l in seq_lens2 if l <= threshold_length]
    num_actually_filtered2 = len(filtered_seq_lens2)
    print(f"Number of filtered sequences 2: {num_actually_filtered2}")
    if num_actually_filtered2 > num_filtered2:
      print(
          f"Warning: Actually filtered more data than expected: "
          f"{num_actually_filtered2} > {num_filtered2} ({num_actually_filtered2 / num_filtered2})"
      )
      assert num_actually_filtered2 / num_filtered2 < 1.001
    else:
      print(
          f"Actually filtered less data than expected: "
          f"{num_actually_filtered2} < {num_filtered2} ({num_actually_filtered2 / num_filtered2})"
      )
      assert num_actually_filtered2 / num_filtered2 > 0.999

    wrongly_filtered2 = [l for l in seq_lens2 if l <= self.max_seq_len1]
    num_wrongly_filtered2 = len(wrongly_filtered2)
    print(
      f"Number of wrongly filtered sequences 2 with max_seq_length {self.max_seq_len1}: "
      f"{num_wrongly_filtered2} ({num_wrongly_filtered2 / num2 * 100:.2f}% of original data)"
    )
    print(
      f"Number of correctly filtered sequences 2 with threshold length {threshold_length}: "
      f"{num_actually_filtered2} ({num_actually_filtered2 / num2 * 100:.2f}% of original data)"
    )
    print(
      f"Number of correctly filtered sequences 1 with max_seq_length {self.max_seq_len1}: "
      f"{num_filtered1} ({num_filtered1 / num1 * 100:.2f}% of original data)"
    )

    with open(self.out_threshold.get_path(), "w+") as f:
      f.write(str(threshold_length))

