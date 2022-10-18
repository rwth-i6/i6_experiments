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


class WordsToCTMJob(Job):
  def __init__(self, stm_path: Path, words_path: Path, dataset_name: str):
    self.words_path = words_path
    self.stm_path = stm_path
    self.dataset_name = dataset_name

    self.out_ctm_file = self.output_path("out.ctm")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.scoring import ScliteHubScoreJob
    ScliteHubScoreJob.create_ctm(
      name=self.dataset_name, ref_stm_filename=self.stm_path.get_path(), source_filename=self.words_path.get_path(),
      target_filename=self.out_ctm_file.get_path())