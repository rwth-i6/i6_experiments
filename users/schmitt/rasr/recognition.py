from sisyphus import *

from i6_core.util import create_executable
from i6_core.rasr.config import build_config_from_mapping
from i6_core.rasr.command import RasrCommand
from i6_core.rasr.flow import FlowNetwork
from i6_core import util

import subprocess
import tempfile
import shutil
from typing import Optional, List, Union
import collections
import gzip
import xml.etree.ElementTree as ET


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
      "run",
      rqmt=self.rqmt,
      args=range(1, self.crp.concurrent + 1),
      resume="run"
    )
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


class RASRDecodingStatisticsJob(RasrCommand, Job):
  def __init__(
          self,
          search_logs: List[Union[str, tk.Path]],
          corpus_duration_hours: float,
  ):
    self.corpus_duration = corpus_duration_hours
    self.search_logs = search_logs

    self.elapsed_time = self.output_var("elapsed_time")
    self.user_time = self.output_var("user_time")
    self.system_time = self.output_var("system_time")
    self.elapsed_rtf = self.output_var("elapsed_rtf")
    self.user_rtf = self.output_var("user_rtf")
    self.system_rtf = self.output_var("system_rtf")
    self.avg_word_ends = self.output_var("avg_word_ends")
    self.avg_trees = self.output_var("avg_trees")
    self.avg_states = self.output_var("avg_states")
    self.recognizer_time = self.output_var("recognizer_time")
    self.recognizer_rtf = self.output_var("recognizer_rtf")
    self.rescoring_time = self.output_var("rescoring_time")
    self.rescoring_rtf = self.output_var("rescoring_rtf")
    self.tf_lm_time = self.output_var("tf_lm_time")
    self.tf_lm_rtf = self.output_var("tf_lm_rtf")
    self.decoding_rtf = self.output_var("decoding_rtf")
    self.ss_statistics = self.output_var("ss_statistics")
    self.seq_ss_statistics = self.output_var("seq_ss_statistics")
    self.eval_statistics = self.output_var("eval_statistics")

    self.rqmt = {"cpu": 1, "mem": 2.0, "time": 0.5}

  def tasks(self):
    yield Task("run", resume="run", rqmt=self.rqmt, mini_task=True)

  def run(self):
    queue_stats = collections.defaultdict(list)

    total_elapsed = 0.0
    total_user = 0.0
    total_system = 0.0
    total_frames = 0
    total_word_ends = 0.0
    total_trees = 0.0
    total_states = 0.0
    recognizer_time = 0.0
    rescoring_time = 0.0
    lm_time = 0.0
    ss_statistics = collections.defaultdict(lambda: (0.0, 0))
    seq_ss_statistics = {}
    eval_statistics = {}

    for path in self.search_logs:
      with gzip.open(path, "rt") as f:
        root = ET.fromstring(f.read())
      host = root.findall("./system-information/name")[0].text
      elapsed = float(root.findall("./timer/elapsed")[0].text)
      user = float(root.findall("./timer/user")[0].text)
      system = float(root.findall("./timer/system")[0].text)
      total_elapsed += elapsed
      total_user += user
      total_system += system

      for layer in root.findall('.//segment/layer[@name="recognizer"]'):
        frames = int(layer.findall('./statistics/frames[@port="features"]')[0].attrib["number"])
        total_frames += frames
        total_word_ends += frames * float(
          layer.findall('./search-space-statistics/statistic[@name="ending words after pruning"]/avg')[0].text
        )
        total_trees += frames * float(
          layer.findall('./search-space-statistics/statistic[@name="trees after  pruning"]/avg')[0].text
        )
        total_states += frames * float(
          layer.findall('./search-space-statistics/statistic[@name="states after pruning"]/avg')[0].text
        )

        recognizer_time += float(layer.findall("./flf-recognizer-time")[0].text)

      for rescore in root.findall(".//segment/flf-push-forward-rescoring-time"):
        rescoring_time += float(rescore.text)

      for lm_total in root.findall(".//fwd-summary/total-run-time"):
        lm_time += float(lm_total.text)

      for seg in root.findall(".//segment"):
        seg_stats = {}
        full_name = seg.attrib["full-name"]
        for sss in seg.findall('./layer[@name="recognizer"]/search-space-statistics/statistic[@type="scalar"]'):
          min_val = float(sss.findtext("./min", default="0"))
          avg_val = float(sss.findtext("./avg", default="0"))
          max_val = float(sss.findtext("./max", default="0"))
          seg_stats[sss.attrib["name"]] = (min_val, avg_val, max_val)
        seq_ss_statistics[full_name] = seg_stats

        features = seg.find('./layer[@name="recognizer"]/statistics/frames[@port="features"]')
        seq_ss_statistics[full_name]["frames"] = int(features.attrib["number"])

        tf_fwd = seg.find(
          './layer[@name="recognizer"]/information[@component="flf-lattice-tool.network.recognizer.feature-extraction.tf-fwd"]'
        )
        if tf_fwd is not None:
          seq_ss_statistics[full_name]["tf_fwd"] = float(tf_fwd.text.strip().split()[-1])
        else:
          seq_ss_statistics[full_name]["tf_fwd"] = 0.0

        eval_statistics[full_name] = {}
        for evaluation in seg.findall(".//evaluation"):
          stat_name = evaluation.attrib["name"]
          alignment = evaluation.find('statistic[@type="alignment"]')
          eval_statistics[full_name][stat_name] = {
            "errors": int(alignment.findtext("edit-operations")),
            "ref-tokens": int(alignment.findtext('count[@event="token"][@source="reference"]')),
            "score": float(alignment.findtext('score[@source="best"]')),
          }

    for s in seq_ss_statistics.values():
      frames = s["frames"]
      for stat, val in s.items():
        if stat == "frames":
          pass
        elif stat == "tf_fwd":
          prev_count, prev_frames = ss_statistics[stat]
          ss_statistics[stat] = (
            prev_count + val,
            (3600.0 * 1000.0 * self.corpus_duration),
          )
        else:
          prev_count, prev_frames = ss_statistics[stat]
          ss_statistics[stat] = (
            prev_count + val[1] * frames,
            prev_frames + frames,
          )
    for s in ss_statistics:
      count, frames = ss_statistics[s]
      ss_statistics[s] = count / frames

    tf_fwd_rtf = ss_statistics.get("tf_fwd", 0)

    self.elapsed_time.set(total_elapsed / 3600.0)
    self.user_time.set(total_user / 3600.0)
    self.system_time.set(total_system / 3600.0)
    self.elapsed_rtf.set(total_elapsed / (3600.0 * self.corpus_duration))
    self.user_rtf.set(total_user / (3600.0 * self.corpus_duration))
    self.system_rtf.set(total_system / (3600.0 * self.corpus_duration))
    self.avg_word_ends.set(total_word_ends / total_frames)
    self.avg_trees.set(total_trees / total_frames)
    self.avg_states.set(total_states / total_frames)
    self.recognizer_time.set(recognizer_time / (3600.0 * 1000.0))
    self.recognizer_rtf.set(recognizer_time / (3600.0 * 1000.0 * self.corpus_duration))
    self.rescoring_time.set(rescoring_time / (3600.0 * 1000.0))
    self.rescoring_rtf.set(rescoring_time / (3600.0 * 1000.0 * self.corpus_duration))
    self.tf_lm_time.set(lm_time / (3600.0 * 1000.0))
    self.tf_lm_rtf.set(lm_time / (3600.0 * 1000.0 * self.corpus_duration))
    self.decoding_rtf.set(
      tf_fwd_rtf + ((recognizer_time + rescoring_time) / (3600.0 * 1000.0 * self.corpus_duration))
    )
    self.ss_statistics.set(dict(ss_statistics.items()))
    self.seq_ss_statistics.set(seq_ss_statistics)
    self.eval_statistics.set(eval_statistics)
