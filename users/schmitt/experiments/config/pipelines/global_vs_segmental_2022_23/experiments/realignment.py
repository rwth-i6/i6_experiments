from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.graph import ReturnnGraph
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_compile_config as get_segmental_compile_config

from i6_experiments.users.schmitt.rasr.realignment import RASRRealignmentParallelJob
from i6_experiments.users.schmitt.alignment.alignment import DumpAlignmentFromTxtJobV2

from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.returnn.training import Checkpoint
from i6_core.corpus.segments import SplitSegmentFileJob

import numpy as np
from typing import Dict, Tuple, Optional

from sisyphus import *


class RasrRealignmentExperiment:
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          variant_params: Dict,
          checkpoint: Checkpoint,
          corpus_key: str,
          length_norm: bool,
          max_segment_len: int,
          base_alias: str,
          length_scale: float,
          time_rqmt: int,
          mem_rqmt: int,
          label_pruning: float,
          concurrent: int = 1,
  ):

    self.concurrent = concurrent

    self.checkpoint = checkpoint
    self.max_segment_len = max_segment_len
    self.length_norm = length_norm
    self.label_pruning = label_pruning
    self.corpus_key = corpus_key
    self.variant_params = variant_params
    self.dependencies = dependencies

    self.returnn_config = get_segmental_compile_config(self.variant_params, length_scale=length_scale)

    self.base_alias = base_alias

    self.time_rqmt = time_rqmt
    self.mem_rqmt = mem_rqmt

  def _get_realignment_config(self) -> Tuple[CommonRasrParameters, RasrConfig]:
    return RasrConfigBuilder.get_realignment_config(
      corpus_path=self.dependencies.corpus_paths[self.corpus_key],
      lexicon_path=self.dependencies.rasr_format_paths.realignment_lexicon_path,
      segment_path=self.dependencies.segment_paths[self.corpus_key],
      length_norm=self.length_norm,
      feature_cache_path=self.dependencies.feature_cache_paths[self.corpus_key],
      reduction_factors=int(np.prod(self.variant_params["config"]["time_red"])),
      blank_label_index=self.dependencies.model_hyperparameters.blank_idx,
      start_label_index=self.dependencies.model_hyperparameters.sos_idx,
      label_pruning=self.label_pruning,
      label_pruning_limit=5000,
      label_recombination_limit=-1,
      label_file=self.dependencies.rasr_format_paths.label_file_path,
      allophone_path=self.dependencies.rasr_format_paths.allophone_path,
      context_size=-1,
      meta_graph_file=ReturnnGraph(self.returnn_config).meta_graph_path,
      state_tying_path=self.dependencies.rasr_format_paths.state_tying_path,
      max_segment_len=self.max_segment_len,
      blank_update_history=True,
      loop_update_history=True)

  def run(self) -> Path:
    realignment_crp, realignment_config = self._get_realignment_config()

    split_segments_job = SplitSegmentFileJob(
      segment_file=self.dependencies.segment_paths[self.corpus_key],
      concurrent=self.concurrent)

    realignment_crp.corpus_config.segments.file = None
    realignment_crp.segment_path = split_segments_job.out_segment_path
    realignment_crp.concurrent = self.concurrent

    realignment_job = RASRRealignmentParallelJob(
      rasr_exe_path=RasrExecutables.am_trainer_path,
      crp=realignment_crp,
      model_checkpoint=self.checkpoint,
      mem_rqmt=self.mem_rqmt,
      time_rqtm=self.time_rqmt,
      am_model_trainer_config=realignment_config,
      blank_allophone_state_idx=self.dependencies.rasr_format_paths.blank_allophone_state_idx,
      use_gpu=True)
    realignment_job.add_alias("%s/realign" % self.base_alias)

    extraction_config, extraction_post_config = RasrConfigBuilder.get_alignment_extraction_config(
      allophone_path=self.dependencies.rasr_format_paths.allophone_path,
      state_tying_path=self.dependencies.rasr_format_paths.state_tying_path,
      lexicon_path=self.dependencies.rasr_format_paths.realignment_lexicon_path,
      corpus_path=self.dependencies.corpus_paths[self.corpus_key],
      segment_path=self.dependencies.segment_paths[self.corpus_key],
      feature_cache_path=self.dependencies.feature_cache_paths[self.corpus_key],
      alignment_cache_path=realignment_job.out_alignment_bundle
    )

    dump_align_from_txt_job = DumpAlignmentFromTxtJobV2(
      rasr_config=extraction_config,
      rasr_post_config=extraction_post_config,
      num_classes=self.dependencies.model_hyperparameters.target_num_labels)
    dump_align_from_txt_job.add_alias("%s/realign_hdf" % self.base_alias)

    return dump_align_from_txt_job.out_hdf_align
