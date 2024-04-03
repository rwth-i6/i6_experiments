from sisyphus import *
from sisyphus import Path

from recipe.i6_core.util import create_executable

import subprocess
import os
from typing import Optional
import shutil

import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)


class AddSilenceToBpeSeqsJob(Job):
  def __init__(
          self,
          bpe_seqs_hdf: Path,
          bpe_vocab: Path,
          phoneme_alignment_cache: Path,
          allophone_file: Path,
          silence_phone: str,
          segment_file: Optional[Path],
          time_rqtm=1,
          mem_rqmt=2,
          returnn_python_exe=None,
          returnn_root=None
  ):
    self.returnn_python_exe = returnn_python_exe
    self.returnn_root = returnn_root

    self.bpe_seqs_hdf = bpe_seqs_hdf
    self.bpe_vocab = bpe_vocab
    self.phoneme_alignment_cache = phoneme_alignment_cache
    self.allophone_file = allophone_file
    self.segment_file = segment_file
    self.silence_phone = silence_phone

    self.time_rqmt = time_rqtm
    self.mem_rqtm = mem_rqmt

    self.out_hdf = self.output_path("out_hdf")
    self.out_vocab = self.output_path("out_vocab")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqtm, "time": self.time_rqmt})

  def run(self):
    command = [
      self.returnn_python_exe.get_path(),
      os.path.join(tools_dir, "add_silence_to_bpe_seqs.py"),
      "--bpe_seqs_hdf", self.bpe_seqs_hdf.get_path(),
      "--bpe_vocab", self.bpe_vocab.get_path(),
      "--phoneme_align_cache", self.phoneme_alignment_cache.get_path(),
      "--allophone_file", self.allophone_file.get_path(),
      "--silence_phone", self.silence_phone,
      "--returnn_root", self.returnn_root.get_path(),
    ]

    if self.segment_file is not None:
      command += ["--segment_file", self.segment_file.get_path()]

    create_executable("rnn.sh", command)
    subprocess.check_call(["./rnn.sh"])
    shutil.move("out_hdf.hdf", self.out_hdf.get_path())
