from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil


class RASRRealignmentJob(Job):
  def __init__(self, rasr_exe_path, am_model_trainer_config, crp, model_checkpoint, blank_allophone_state_idx,
               time_rqtm=1, mem_rqmt=2):
    self.am_model_trainer_config = am_model_trainer_config
    self.crp = crp
    self.rasr_exe_path = rasr_exe_path
    self.model_checkpoint = model_checkpoint
    self.blank_allophone_state_idx = blank_allophone_state_idx

    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqtm

    self.out_alignment = self.output_path("alignment.cache.1")
    self.out_alignment_txt = self.output_path("best_traces")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt, "gpu": 1})

  def run(self):
    config, post_config = build_config_from_mapping(self.crp, {
      "corpus": "acoustic-model-trainer.corpus",
      "lexicon": "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.lexicon",
      "acoustic_model": "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"
    })
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction._update(self.am_model_trainer_config)
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment_cache.path = self.out_alignment
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment.label_scorer.loader.saved_model_file = self.model_checkpoint.ckpt_path
    config.acoustic_model_trainer.action = "dry"

    tmp_file = tempfile.NamedTemporaryFile()
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment.aligner.dump_alignment.channel = "alignment"
    config.acoustic_model_trainer.channels.alignment.append = False
    config.acoustic_model_trainer.channels.alignment.compressed = False
    config.acoustic_model_trainer.channels.alignment.file = tmp_file.name
    config.acoustic_model_trainer.channels.alignment.unbuffered = False

    config["*"].blank_allophone_state_idx = self.blank_allophone_state_idx

    RasrCommand.write_config(config, post_config, "rasr.config")
    command = [
      self.rasr_exe_path.get_path(),
      "--config", "rasr.config", "--*.LOGFILE=sprint.log",
    ]

    create_executable("run.sh", command)
    subprocess.check_call(["./run.sh"])

    shutil.copy(tmp_file.name, self.out_alignment_txt.get_path())

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("time_rqtm")
    kwargs.pop("mem_rqmt")
    return super().hash(kwargs)