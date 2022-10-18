from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil
import os
import json

from recipe.i6_experiments.users.schmitt.experiments.swb.transducer import config as config_mod
tools_dir = os.path.join(os.path.dirname(os.path.abspath(config_mod.__file__)), "tools")


class CalcSearchErrorJob(Job):
  def __init__(
    self, returnn_config, rasr_config, rasr_nn_trainer_exe, segment_file, ref_targets, search_targets, blank_idx,
    label_name, model_type, max_seg_len, length_norm, returnn_python_exe=None, returnn_root=None):
    self.blank_idx = blank_idx
    self.rasr_nn_trainer_exe = rasr_nn_trainer_exe
    self.rasr_config = rasr_config
    self.search_targets = search_targets
    self.ref_targets = ref_targets
    self.segment_file = segment_file
    self.label_name = label_name
    self.model_type = model_type
    self.max_seg_len = max_seg_len
    self.length_norm = length_norm
    self.returnn_config = returnn_config
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.out_search_errors = self.output_path("search_errors")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 6, "time": 3, "gpu": 1})

  def run(self):
    self.returnn_config.write("returnn.config")

    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "calc_search_error.py"),
      "returnn.config", "--rasr_nn_trainer_exe", self.rasr_nn_trainer_exe.get_path(),
      "--segment_file", self.segment_file.get_path(), "--blank_idx", str(self.blank_idx),
      "--rasr_config_path", self.rasr_config.get_path(), "--ref_targets", self.ref_targets.get_path(),
      "--search_targets", self.search_targets.get_path(), "--label_name", self.label_name, "--model_type", self.model_type,
      "--max_seg_len", str(self.max_seg_len),
      "--returnn_root", self.returnn_root
    ]

    if self.length_norm:
      command.append("--length_norm")

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("search_errors", self.out_search_errors)


class DumpAttentionWeightsJob(Job):
  def __init__(
    self, returnn_config, rasr_config, rasr_nn_trainer_exe, seq_tag, hdf_targets, blank_idx, label_name, model_type,
    concat_seqs=False, concat_hdf=False, returnn_python_exe=None, returnn_root=None):
    self.model_type = model_type
    self.label_name = label_name
    self.blank_idx = blank_idx
    self.rasr_nn_trainer_exe = rasr_nn_trainer_exe
    self.rasr_config = rasr_config
    self.hdf_targets = hdf_targets
    self.seq_tag = seq_tag
    self.concat_seqs = concat_seqs
    self.concat_hdf = concat_hdf
    self.returnn_config = returnn_config
    self.returnn_python_exe = (returnn_python_exe if returnn_python_exe is not None else gs.RETURNN_PYTHON_EXE)
    self.returnn_root = (returnn_root if returnn_root is not None else gs.RETURNN_ROOT)

    self.out_data = self.output_path("out_data.npz")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 6, "time": 1})

  def run(self):
    self.returnn_config.write("returnn.config")

    with open("seq_file", "w+") as f:
      f.write(self.seq_tag + "\n")

    command = [
      self.returnn_python_exe,
      os.path.join(tools_dir, "dump_attention_weights.py"),
      "returnn.config", "--rasr_nn_trainer_exe", self.rasr_nn_trainer_exe.get_path(),
      "--segment_file", "seq_file",
      "--rasr_config_path", self.rasr_config.get_path(), "--hdf_targets", self.hdf_targets.get_path(),
      "--blank_idx", str(self.blank_idx), "--label_name", self.label_name, "--model_type", self.model_type,
      "--returnn_root", self.returnn_root
    ]

    if self.concat_seqs:
      command += ["--concat_seqs"]
    if self.concat_hdf:
      command += ["--concat_hdf"]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])

    shutil.move("data.npz", self.out_data)

  @classmethod
  def hash(cls, kwargs):
    if not kwargs["concat_seqs"]:
      kwargs.pop("concat_seqs")
      if not kwargs["concat_hdf"]:
        kwargs.pop("concat_hdf")
    return super().hash(kwargs)
