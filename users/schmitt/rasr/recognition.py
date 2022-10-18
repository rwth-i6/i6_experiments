from sisyphus import *

from recipe.i6_core.util import create_executable
from recipe.i6_core.rasr.config import build_config_from_mapping
from recipe.i6_core.rasr.command import RasrCommand

import subprocess
import tempfile
import shutil


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