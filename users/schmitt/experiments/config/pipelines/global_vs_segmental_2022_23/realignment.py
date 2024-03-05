from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple

from i6_core.returnn.compile import CompileTFGraphJob
from i6_core.rasr.crp import CommonRasrParameters
from i6_core.rasr.config import RasrConfig
from i6_core.corpus.segments import SplitSegmentFileJob
from i6_core.features.common import samples_flow
from i6_core.text.processing import WriteToTextFileJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.returnn.training import Checkpoint

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutablesNew
from i6_experiments.users.schmitt.rasr.realignment import RASRRealignmentParallelJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import RasrDecodingExperiment


class RasrRealignmentExperiment(RasrDecodingExperiment):
  def __init__(
          self,
          alias: str,
          **kwargs,
  ):
    super().__init__(alias=alias, **kwargs)

    self.alias = alias
    self.alias += "/rasr_realignment/max-seg-len-%d" % self.max_segment_len

    if self.lm_opts is not None:
      self.alias += "/lm-%s_scale-%f" % (self.lm_opts["type"], self.lm_opts["scale"])
    else:
      self.alias += "/no_lm"

    if self.lm_lookahead_opts is not None:
      self.alias += "/lm-lookahead-scale-%f" % self.lm_lookahead_opts["scale"]
    else:
      self.alias += "/wo-lm-lookahead"

    if self.ilm_correction_opts is not None:
      self.alias += "/ilm_correction_scale-%f" % self.ilm_correction_opts["scale"]
      if self.ilm_correction_opts["correct_eos"]:
        self.alias += "_correct_eos"
      else:
        self.alias += "_wo_correct_eos"
    else:
      self.alias += "/wo_ilm_correction"

  def _get_lexicon_path(self) -> Path:
    return self.config_builder.variant_params["dependencies"].rasr_format_paths.tfrnn_lm_bpe_phoneme_lexicon_path

  def _get_realignment_config(self) -> Tuple[CommonRasrParameters, RasrConfig]:
    return RasrConfigBuilder.get_realignment_config(
      corpus_path=self.config_builder.variant_params["dataset"]["corpus"].corpus_paths_wav[self.corpus_key],
      segment_path=self._get_segment_path(),
      lexicon_path=self._get_lexicon_path(),
      feature_cache_path=self.config_builder.variant_params["dataset"]["corpus"].oggzip_paths[self.corpus_key],
      feature_extraction_file="feature.flow",
      label_pruning=self.label_pruning,
      label_pruning_limit=self.label_pruning_limit,
      length_norm=self.length_norm,
      max_segment_len=self.max_segment_len,
      loop_update_history=True,
      blank_update_history=True,
      label_file=self.config_builder.variant_params["dependencies"].rasr_format_paths.label_file_path,
      start_label_index=self.config_builder.variant_params["dependencies"].model_hyperparameters.sos_idx,
      blank_label_index=self.config_builder.variant_params["dependencies"].model_hyperparameters.blank_idx,
      label_recombination_limit=-1,
      reduction_factors=self.reduction_factor,
      reduction_subtrahend=self.reduction_subtrahend,
      meta_graph_path=self._get_returnn_graph(),
      native_lstm2_so_path=self.native_lstm2_so_path,
      allophone_path=self.config_builder.variant_params["dependencies"].rasr_format_paths.allophone_path,
      state_tying_path=self.config_builder.variant_params["dependencies"].rasr_format_paths.state_tying_path,
      context_size=-1,
    )

  def do_realignment(self):
    realignment_crp, realignment_config = self._get_realignment_config()

    if self.concurrent > 1:
      split_segments_job = SplitSegmentFileJob(
        segment_file=self._get_segment_path(),
        concurrent=self.concurrent
      )

      realignment_crp.corpus_config.segments.file = None
      realignment_crp.segment_path = split_segments_job.out_segment_path
      realignment_crp.concurrent = self.concurrent

    rasr_realignment_job = RASRRealignmentParallelJob(
      rasr_exe_path=RasrExecutablesNew.am_trainer_path,
      am_model_trainer_config=realignment_config,
      crp=realignment_crp,
      feature_flow=samples_flow(dc_detection=False, scale_input=3.0517578125e-05, input_options={"block-size": "1"}),
      model_checkpoint=self.checkpoint,
      mem_rqmt=self.search_rqmt.get("mem", 4),
      time_rqmt=self.search_rqmt.get("time", 1),
      use_gpu=self.search_rqmt.get("gpu", 1) > 0,
      blank_allophone_state_idx=self.config_builder.variant_params["dependencies"].rasr_format_paths.blank_allophone_state_idx,
    )
    rasr_realignment_job.add_alias("%s/realign_%s" % (self.alias, self.corpus_key))
    tk.register_output(rasr_realignment_job.get_one_alias(), rasr_realignment_job.out_alignment_bundle)
