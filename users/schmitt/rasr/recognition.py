from sisyphus import *

from i6_core.util import create_executable
from i6_core.rasr.config import build_config_from_mapping
from i6_core.rasr.command import RasrCommand
from i6_core.rasr.flow import FlowNetwork
from i6_core import util

import subprocess
import tempfile
import shutil
from typing import Optional


class RASRDecodingJob(Job):
  def __init__(self, rasr_exe_path, flf_lattice_tool_config, crp, model_checkpoint, dump_best_trace,
               time_rqmt=1, mem_rqmt=2, gpu_rqmt=1):
    self.crp = crp
    self.rasr_exe_path = rasr_exe_path
    self.flf_lattice_tool_config = flf_lattice_tool_config
    self.model_checkpoint = model_checkpoint
    self.dump_best_trace = dump_best_trace

    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqmt
    self.gpu_rqmt = gpu_rqmt

    self.out_lattice = self.output_path("lattice.cache.1")
    self.out_best_traces = self.output_path("best_traces")
    self.out_log = self.output_path("sprint.log")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": self.mem_rqmt, "time": self.time_rqmt, "gpu": self.gpu_rqmt})

  def run(self):
    config, post_config = build_config_from_mapping(self.crp, {
      "corpus": "flf-lattice-tool.corpus",
      "lexicon": "flf-lattice-tool.lexicon",
      "recognizer": "flf-lattice-tool.network.recognizer"
    })
    config.flf_lattice_tool._update(self.flf_lattice_tool_config)
    config.flf_lattice_tool.network.archive_writer.path = self.out_lattice
    config.flf_lattice_tool.network.recognizer.label_scorer.loader.saved_model_file = self.model_checkpoint.ckpt_path

    tmp_file = None
    if self.dump_best_trace:
      tmp_file = tempfile.NamedTemporaryFile()
      config.flf_lattice_tool.network.recognizer.recognizer.dump_alignment.channel = "alignment"
      config.flf_lattice_tool.channels.alignment.append = False
      config.flf_lattice_tool.channels.alignment.compressed = False
      config.flf_lattice_tool.channels.alignment.file = tmp_file.name
      config.flf_lattice_tool.channels.alignment.unbuffered = False

    RasrCommand.write_config(config, post_config, "rasr.config")
    command = [
      self.rasr_exe_path.get_path(),
      "--config", "rasr.config", "--*.LOGFILE=sprint.log",
    ]

    create_executable("run.sh", command)
    subprocess.check_call(["./run.sh"])

    if tmp_file is not None:
      shutil.copy(tmp_file.name, self.out_best_traces.get_path())

    shutil.move("sprint.log", self.out_log.get_path())

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("time_rqmt")
    kwargs.pop("mem_rqmt")
    kwargs.pop("gpu_rqmt")
    return super().hash(kwargs)


class RASRDecodingJobParallel(RasrCommand, Job):
  def __init__(
          self,
          rasr_exe_path,
          flf_lattice_tool_config,
          crp,
          model_checkpoint,
          dump_best_trace,
          time_rqmt=1,
          mem_rqmt=2,
          use_gpu=True,
          feature_flow: Optional[FlowNetwork] = None,
  ):
    self.crp = crp
    self.rasr_exe_path = rasr_exe_path
    self.flf_lattice_tool_config = flf_lattice_tool_config
    self.feature_flow = feature_flow
    self.model_checkpoint = model_checkpoint
    self.dump_best_trace = dump_best_trace

    self.mem_rqmt = mem_rqmt
    self.time_rqmt = time_rqmt
    self.use_gpu = use_gpu

    self.out_log_file = self.log_file_output_path("lattice", crp, True)
    self.out_single_lattice_caches = dict(
      (task_id, self.output_path("lattice.cache.%d" % task_id, cached=True))
      for task_id in range(1, crp.concurrent + 1)
    )
    self.out_lattice_bundle = self.output_path("lattice.bundle", cached=True)
    self.out_lattice_path = util.MultiOutputPath(
      self, "lattice.cache.$(TASK)", self.out_single_lattice_caches, cached=True
    )

    self.rqmt = {
      "cpu": 3,
      "mem": self.mem_rqmt, "time": self.time_rqmt, "gpu": 1 if use_gpu else 0}

    # self.out_lattice = self.output_path("lattice.cache.1")
    self.out_best_traces = self.output_path("best_traces")
    # self.out_log = self.output_path("sprint.log")

  def tasks(self):
    yield Task("create_files", mini_task=True)
    yield Task(
      "run", rqmt=self.rqmt, args=range(1, self.crp.concurrent + 1), resume="run")
    if self.dump_best_trace:
      yield Task("create_trace_file", mini_task=True)

  def create_files(self):
    config, post_config = build_config_from_mapping(self.crp, {
      "corpus": "flf-lattice-tool.corpus",
      "lexicon": "flf-lattice-tool.lexicon",
      "recognizer": "flf-lattice-tool.network.recognizer"
    }, parallelize=True)
    config.flf_lattice_tool._update(self.flf_lattice_tool_config)
    config.flf_lattice_tool.network.archive_writer.path = "lattice.cache.$(TASK)"
    config.flf_lattice_tool.network.recognizer.label_scorer.loader.saved_model_file = self.model_checkpoint.ckpt_path

    # specifies that RASR processes the segments in the same order as the given segment file
    config.flf_lattice_tool.corpus.segment_order = config.flf_lattice_tool.corpus.segments.file

    if self.dump_best_trace:
      config.flf_lattice_tool.network.recognizer.recognizer.dump_alignment.channel = "alignment"
      config.flf_lattice_tool.channels.alignment.append = False
      config.flf_lattice_tool.channels.alignment.compressed = False
      config.flf_lattice_tool.channels.alignment.file = "best_traces.$(TASK)"
      config.flf_lattice_tool.channels.alignment.unbuffered = False

    RasrCommand.write_config(config, post_config, "rasr.config")
    if self.feature_flow is not None:
      self.feature_flow.write_to_file("feature.flow")
    util.write_paths_to_file(
      self.out_lattice_bundle, self.out_single_lattice_caches.values()
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
      "lattice.cache.%d" % task_id, self.out_single_lattice_caches[task_id].get_path())

  def cleanup_before_run(self, cmd, retry, task_id, *args):
    util.backup_if_exists("lattice.log.%d" % task_id)
    util.delete_if_exists("lattice.cache.%d" % task_id)

  def create_trace_file(self):
    with open(self.out_best_traces.get_path(), "w+") as f1:
      for i in range(1, self.crp.concurrent + 1):
        with open("best_traces.%d" % i, "r") as f2:
          for line in f2:
            f1.write(line)

  @classmethod
  def hash(cls, kwargs):
    kwargs.pop("time_rqmt")
    kwargs.pop("mem_rqmt")
    return super().hash(kwargs)
