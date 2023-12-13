from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand
from i6_core.returnn.config import ReturnnConfig
from i6_core.returnn.forward import ReturnnForwardJob

from sisyphus import Path

import subprocess
import tempfile
import shutil
import os
import json

import recipe.i6_private.users.gruev.tools as tools_mod
# import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)

class AlignmentStatisticsJob(Job):
  def __init__(self, alignment, seq_list_filter_file=None, blank_idx=0, silence_idx=None,
               time_rqmt=2, returnn_python_exe=None,
               returnn_root=None):
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.alignment = alignment
    self.seq_list_filter_file = seq_list_filter_file
    self.blank_idx = blank_idx
    self.silence_idx = silence_idx
    self.out_statistics = self.output_path("statistics")
    self.out_sil_hist = self.output_path("sil_histogram.pdf")
    self.out_non_sil_hist = self.output_path("non_sil_histogram.pdf")
    self.out_label_dep_stats = self.output_path("label_dep_mean_lens")
    # self.out_label_dep_vars = self.output_path("label_dep_mean_vars")
    self.out_label_dep_stats_var = self.output_var("label_dep_mean_lens_var", pickle=True)
    # self.out_label_dep_vars_var = self.output_var("label_dep_mean_vars_var", pickle=True)
    self.out_mean_non_sil_len = self.output_path("mean_non_sil_len")
    self.out_mean_non_sil_len_var = self.output_var("mean_non_sil_len_var")
    self.out_95_percentile_var = self.output_var("percentile_95")
    self.out_90_percentile_var = self.output_var("percentile_90")
    self.out_99_percentile_var = self.output_var("percentile_99")

    self.time_rqmt = time_rqmt

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "schmitt_segment_statistics.py"),
      self.alignment.get_path(),
      "--blank-idx", str(self.blank_idx), "--sil-idx", str(self.silence_idx),
      "--returnn-root", self.returnn_root.get_path()
    ]

    if self.seq_list_filter_file:
      command += ["--seq-list-filter-file", str(self.seq_list_filter_file)]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    with open("label_dep_mean_lens", "r") as f:
      label_dep_means = json.load(f)
      label_dep_means = {int(k): v for k, v in label_dep_means.items()}
      label_dep_means = [label_dep_means[idx] for idx in range(len(label_dep_means)) if idx > 1]

    # with open("label_dep_vars", "r") as f:
    #   label_dep_vars = json.load(f)
    #   label_dep_vars = {int(k): v for k, v in label_dep_vars.items()}
    #   label_dep_vars = [label_dep_vars[idx] for idx in range(len(label_dep_vars))]

    with open("mean_non_sil_len", "r") as f:
      self.out_mean_non_sil_len_var.set(float(f.read()))

    # set percentiles
    with open("percentile_90", "r") as f:
      self.out_90_percentile_var.set(int(float(f.read())))
    with open("percentile_95", "r") as f:
      self.out_95_percentile_var.set(int(float(f.read())))
    with open("percentile_99", "r") as f:
      self.out_99_percentile_var.set(int(float(f.read())))

    self.out_label_dep_stats_var.set(label_dep_means)
    # self.out_label_dep_vars_var.set(label_dep_vars)

    shutil.move("statistics", self.out_statistics.get_path())
    shutil.move("sil_histogram.pdf", self.out_sil_hist.get_path())
    shutil.move("non_sil_histogram.pdf", self.out_non_sil_hist.get_path())
    shutil.move("label_dep_mean_lens", self.out_label_dep_stats.get_path())
    # shutil.move("label_dep_vars", self.out_label_dep_vars.get_path())
    shutil.move("mean_non_sil_len", self.out_mean_non_sil_len.get_path())
