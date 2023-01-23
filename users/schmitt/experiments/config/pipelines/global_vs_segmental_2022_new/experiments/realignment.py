from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.returnn.graph import ReturnnGraph
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.returnn.config.seg import get_compile_config as get_segmental_compile_config

from i6_private.users.schmitt.returnn.tools import RASRRealignmentJob, DumpAlignmentFromTxtJobNew

from i6_core.rasr.config import RasrConfig
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.returnn.training import Checkpoint

import numpy as np
from typing import Dict, Tuple, Optional

from sisyphus import *


class RasrRealignmentExperiment:
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          variant_params: Dict,
          checkpoint: Checkpoint,
          epoch: int,
          corpus_key: str,
          length_norm: bool,
          max_segment_len: int
  ):

    self.checkpoint = checkpoint
    self.max_segment_len = max_segment_len
    self.length_norm = length_norm
    self.corpus_key = corpus_key
    self.epoch = epoch
    self.variant_params = variant_params
    self.dependencies = dependencies

    self.base_alias = "%s/rasr_realign_limit12_pruning12.0_vit-recomb_neural-length_max-seg-len-20_length-scale-1.0/epoch_%d" % self.epoch
    self.returnn_config = get_segmental_compile_config(self.variant_params, length_scale=1.0)

  def _get_realignment_config(self) -> Tuple[CommonRasrParameters, RasrConfig]:
    return RasrConfigBuilder.get_realignment_config(
      corpus_path=self.dependencies.corpus_paths[self.corpus_key],
      lexicon_path=self.dependencies.rasr_format_paths.lexicon_path,
      segment_path=self.dependencies.segment_paths[self.corpus_key],
      length_norm=self.length_norm,
      feature_cache_path=self.dependencies.feature_cache_paths[self.corpus_key],
      reduction_factors=int(np.prod(self.variant_params["config"]["time_red"])),
      blank_label_index=self.dependencies.model_hyperparameters.blank_idx,
      start_label_index=self.dependencies.model_hyperparameters.blank_idx,
      label_pruning=12.0,
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

  def get_ctm_path(self) -> Path:
    realignment_crp, realignment_config = self._get_realignment_config()

    realignment_job = RASRRealignmentJob(
      rasr_exe_path=RasrExecutables.am_trainer_path,
      crp=realignment_crp,
      model_checkpoint=self.checkpoint,
      mem_rqmt=8,
      time_rqtm=8,
      am_model_trainer_config=realignment_config,
      blank_allophone_state_idx=self.dependencies.rasr_format_paths.blank_allophone_state_idx)
    realignment_job.add_alias("%s/rasr-realignment_%s" % (self.base_alias, self.corpus_key))

    extraction_config, extraction_post_config = RasrConfigBuilder.get_alignment_extraction_config(
      allophone_path=self.dependencies.rasr_format_paths.allophone_path,
      state_tying_path=self.dependencies.rasr_format_paths.state_tying_path,
      lexicon_path=self.dependencies.rasr_format_paths.lexicon_path,
      corpus_path=self.dependencies.corpus_paths[self.corpus_key],
      segment_path=self.dependencies.segment_paths[self.corpus_key],
      feature_cache_path=self.dependencies.feature_cache_paths[self.corpus_key],
      alignment_cache_path=realignment_job.out_alignment
    )

    dump_align_from_txt_job = DumpAlignmentFromTxtJobNew(
      rasr_config=extraction_config,
      rasr_post_config=extraction_post_config,
      num_classes=self.dependencies.model_hyperparameters.blank_idx)
    dump_align_from_txt_job.add_alias("%s/rasr-realignment_%s_best-traces" % (self.base_alias, self.corpus_key))
    tk.register_output(dump_align_from_txt_job.get_one_alias(), dump_align_from_txt_job.out_hdf_align)

    return dump_align_from_txt_job.out_hdf_align
