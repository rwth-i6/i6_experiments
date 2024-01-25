from sisyphus import *

from i6_core.util import create_executable
from i6_core.rasr.config import build_config_from_mapping
from i6_core.rasr.command import RasrCommand
from i6_core import util
from i6_core.rasr.flow import FlowNetwork

import subprocess
import tempfile
import shutil
from typing import Optional


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


class RASRRealignmentParallelJob(RasrCommand, Job):
  def __init__(
          self,
          rasr_exe_path,
          am_model_trainer_config,
          crp,
          model_checkpoint,
          blank_allophone_state_idx,
          time_rqmt=1,
          mem_rqmt=2,
          use_gpu=False,
          feature_flow: Optional[FlowNetwork] = None,
  ):
    self.am_model_trainer_config = am_model_trainer_config
    self.crp = crp
    self.rasr_exe_path = rasr_exe_path
    self.model_checkpoint = model_checkpoint
    self.blank_allophone_state_idx = blank_allophone_state_idx
    self.feature_flow = feature_flow

    self.use_gpu = use_gpu
    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqmt

    self.out_log_file = self.log_file_output_path("alignment", crp, True)
    self.out_single_alignment_caches = dict(
      (task_id, self.output_path("alignment.cache.%d" % task_id, cached=True))
      for task_id in range(1, crp.concurrent + 1)
    )
    self.out_alignment_bundle = self.output_path("alignment.bundle", cached=True)
    self.out_alignment_path = util.MultiOutputPath(
      self, "alignment.cache.$(TASK)", self.out_single_alignment_caches, cached=True
    )

    self.rqmt = {
      "cpu": 3,
      "mem": self.mem_rqmt, "time": self.time_rqmt, "gpu": 1 if use_gpu else 0}

    # self.out_alignment = self.output_path("alignment.cache.1")
    # self.out_alignment_txt = self.output_path("best_traces")

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task(
      "run", rqmt=self.rqmt,
      args=range(1, self.crp.concurrent + 1))

  def create_files(self):
    config, post_config = build_config_from_mapping(self.crp, {
      "corpus": "acoustic-model-trainer.corpus",
      "lexicon": "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.lexicon",
      "acoustic_model": "acoustic-model-trainer.aligning-feature-extractor.feature-extraction.alignment.model-combination.acoustic-model"
    }, parallelize=True)
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction._update(self.am_model_trainer_config)
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment_cache.path = "alignment.cache.$(TASK)"
    config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment.label_scorer.loader.saved_model_file = self.model_checkpoint.ckpt_path
    config.acoustic_model_trainer.action = "dry"

    # tmp_file = tempfile.NamedTemporaryFile()
    # config.acoustic_model_trainer.aligning_feature_extractor.feature_extraction.alignment.aligner.dump_alignment.channel = "alignment"
    # config.acoustic_model_trainer.channels.alignment.append = False
    # config.acoustic_model_trainer.channels.alignment.compressed = False
    # config.acoustic_model_trainer.channels.alignment.file = tmp_file.name
    # config.acoustic_model_trainer.channels.alignment.unbuffered = False

    config["*"].blank_allophone_state_idx = self.blank_allophone_state_idx

    RasrCommand.write_config(config, post_config, "rasr.config")
    if self.feature_flow is not None:
      self.feature_flow.write_to_file("feature.flow")
    util.write_paths_to_file(
      self.out_alignment_bundle, self.out_single_alignment_caches.values()
    )
    extra_code = (
      'export TF_DEVICE="{0}"'.format("gpu" if self.use_gpu else "cpu")
    )
    # sometimes crash without this
    if not self.use_gpu:
      extra_code += "\nexport CUDA_VISIBLE_DEVICES="
    extra_code += "\nexport OMP_NUM_THREADS=%i" % self.rqmt["cpu"]
    RasrCommand.write_run_script(self.rasr_exe_path.get_path(), "rasr.config", extra_code=extra_code)

  def run(self, task_id):
    self.run_script(task_id, self.out_log_file[task_id], use_tmp_dir=False)
    shutil.move(
      "alignment.cache.%d" % task_id, self.out_single_alignment_caches[task_id].get_path()
    )
    # command = [
    #   self.rasr_exe_path.get_path(),
    #   "--config", "rasr.config", "--*.LOGFILE=sprint.log",
    # ]
    #
    # create_executable("run.sh", command)
    # subprocess.check_call(["./run.sh"])

    # shutil.copy(tmp_file.name, self.out_alignment_txt.get_path())

  def cleanup_before_run(self, cmd, retry, task_id, *args):
    util.backup_if_exists("alignment.log.%d" % task_id)
    util.delete_if_exists("alignment.cache.%d" % task_id)

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("time_rqmt")
    kwargs.pop("mem_rqmt")
    return super().hash(kwargs)