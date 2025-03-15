from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import string
import copy

from i6_core.returnn import ReturnnConfig as RF
from i6_core.util import instanciate_delayed
from i6_experiments.common.setups.serialization import PartialImport as PI
from i6_experiments.users.schmitt.nn.util import DelayedCodeWrapper
from sisyphus.hash import sis_hash_helper

import subprocess
import tempfile
import shutil
import os
import re
import numpy as np
from sisyphus.toolkit import Variable

from typing import Iterator, Optional, List

import recipe.i6_experiments.users.schmitt.tools as tools_mod
tools_dir = os.path.dirname(tools_mod.__file__)


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
      if type(self.seqs_to_skip) == Variable:
        seqs_to_skip = self.seqs_to_skip.get()
      else:
        assert type(self.seqs_to_skip) == list or type(self.seqs_to_skip) == tuple
        seqs_to_skip = self.seqs_to_skip
      with open(self.seq_file.get_path(), "r") as src:
        with open(self.out_seqs_file.get_path(), "w+") as dst:
          for line in src:
            if line.strip() not in seqs_to_skip:
              dst.write(line)


class GetLearningRateFromFileJob(Job):
  def __init__(
          self,
          lr_file_path: Path,
          epoch: Optional[int] = None
  ):
    """

    :param lr_file_path:
    :param epoch: epoch for which to get the lr from the lr file. If `None`, use last epoch.
    """
    self.lr_file_path = lr_file_path
    self.epoch = epoch

    self.out_last_lr = self.output_var("out_last_lr")

  def tasks(self) -> Iterator[Task]:
    yield Task("run", rqmt={"cpu": 1}, mini_task=True)

  def run(self):
    with open(self.lr_file_path.get_path(), "r") as lr_file:
      file_contents = lr_file.read().strip()

    learning_rates = re.findall("learningRate=(0\.\d+)", file_contents)
    learning_rate = float(learning_rates[self.epoch - 1 if type(self.epoch) == int else -1])

    self.out_last_lr.set(learning_rate)


class CombineNpyFilesJob(Job):
  def __init__(
          self,
          npy_files: List[Path],
          epoch: Optional[int] = None
  ):
    """

    :param lr_file_path:
    :param epoch: epoch for which to get the lr from the lr file. If `None`, use last epoch.
    """
    self.npy_files = npy_files

    self.out_file = self.output_path("combined.npy")

  def tasks(self) -> Iterator[Task]:
    yield Task("run", rqmt={"cpu": 1}, mini_task=True)

  def run(self):
    result = {}

    for i, npy_file_path in enumerate(self.npy_files):
      npy_object = np.load(npy_file_path.get_path(), allow_pickle=True)[()][0]

      result[i] = npy_object

    result = np.array(result)

    np.save(self.out_file.get_path(), result)


class PartialImportCustom(PI):
  def get(self) -> str:
    arguments = {**self.hashed_arguments}
    arguments.update(self.unhashed_arguments)
    print(arguments)
    return string.Template(self.TEMPLATE).substitute(
      {
        "KWARGS": str(instanciate_delayed(arguments)),
        "IMPORT_PATH": self.module,
        "IMPORT_NAME": self.object_name,
        "OBJECT_NAME": self.import_as if self.import_as is not None else self.object_name,
      }
    )


class ReturnnConfigCustom(RF):
  def __init__(
          self,
          config,
          post_config=None,
          staged_network_dict=None,
          *,
          python_prolog=None,
          python_prolog_hash=None,
          python_epilog="",
          python_epilog_hash=None,
          hash_full_python_code=False,
          sort_config=True,
          pprint_kwargs=None,
          black_formatting=True,
  ):
    if python_prolog_hash is None and python_prolog is not None:
      python_prolog_hash = []

    super().__init__(
      config=config,
      post_config=post_config,
      staged_network_dict=staged_network_dict,
      python_prolog=python_prolog,
      python_prolog_hash=python_prolog_hash,
      python_epilog=python_epilog,
      python_epilog_hash=python_epilog_hash,
      hash_full_python_code=hash_full_python_code,
      sort_config=sort_config,
      pprint_kwargs=pprint_kwargs,
      black_formatting=black_formatting,
    )

    if self.python_prolog_hash == []:
      self.python_prolog_hash = None

  def _sis_hash(self):
    conf = copy.deepcopy(self.config)
    if "preload_from_files" in conf:
      for v in conf["preload_from_files"].values():
        if "filename" in v and isinstance(v["filename"], DelayedCodeWrapper):
          v["filename"] = v["filename"].args[0]
    h = {
      "returnn_config": conf,
      "python_epilog_hash": self.python_epilog_hash,
      "python_prolog_hash": self.python_prolog_hash,
    }
    if self.staged_network_dict:
      h["returnn_networks"] = self.staged_network_dict

    return sis_hash_helper(h)
