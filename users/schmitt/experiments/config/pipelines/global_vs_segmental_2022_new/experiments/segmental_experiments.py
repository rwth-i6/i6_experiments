from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import SegmentalLabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.experiments.recognition import SegmentalReturnnDecodingExperiment, RasrDecodingExperiment, DecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.experiments.search_errors import SegmentalSearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.experiments.visualization import AlignmentComparer, SegmentalAttentionWeightsPlotter
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.returnn.config.seg import get_recog_config as get_segmental_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.returnn.config.seg import get_compile_config as get_segmental_compile_config

from i6_core.returnn.training import Checkpoint

from . import seq_tags_for_analysis

from typing import Dict, Optional
from abc import ABC, abstractmethod


class SearchPipeline(ABC):
  """
  Pipeline consists of decoding on the DEV set and then optional calculation of search errors, visualization of
  attention weights and visualization of alignment comparison on the CV set.
  """
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          variant_params: Dict,
          variant_name: str,
          checkpoint: Checkpoint,
          epoch: int,
          calc_search_errors: bool,
          plot_att_weights: bool,
          compare_to_ground_truth_align: bool):

    self.compare_to_ground_truth_align = compare_to_ground_truth_align
    self.plot_att_weights = plot_att_weights
    self.calc_search_errors = calc_search_errors
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.variant_name = variant_name
    self.checkpoint = checkpoint
    self.epoch = epoch

  @abstractmethod
  def get_decoding_experiment(self, corpus_key: str, dump_best_traces: bool) -> DecodingExperiment:
    pass

  def decode(self, corpus_key: str, dump_best_traces: bool):
    decoding_exp = self.get_decoding_experiment(corpus_key=corpus_key, dump_best_traces=dump_best_traces)
    decoding_exp.run_eval()
    return decoding_exp.best_traces

  def run(self):
    self.decode(corpus_key="dev", dump_best_traces=False)

    if self.calc_search_errors or self.plot_att_weights or self.compare_to_ground_truth_align:
      cv_search_aligns = self.decode(corpus_key="cv", dump_best_traces=True)

      if self.calc_search_errors:
        # calculate search errors on CV
        SegmentalSearchErrorExperiment(
          dependencies=self.dependencies, checkpoint=self.checkpoint, variant_name=self.variant_name,
          epoch=self.epoch, variant_params=self.variant_params, search_targets=cv_search_aligns)

      if self.plot_att_weights:
        for hdf_target_path, hdf_alias in zip(
          [self.dependencies.alignment_paths["cv"], cv_search_aligns], ["ground-truth", "search"]
        ):
          for seq_tag in seq_tags_for_analysis:
            SegmentalAttentionWeightsPlotter(
              dependencies=self.dependencies, variant_name=self.variant_name,
              variant_params=self.variant_params, checkpoint=self.checkpoint, corpus_key="cv", seq_tag=seq_tag,
              hdf_target_path=hdf_target_path, hdf_alias=hdf_alias)

      if self.compare_to_ground_truth_align:
        for seq_tag in seq_tags_for_analysis:
          AlignmentComparer(
            variant_name=self.variant_name, hdf_align_path1=self.dependencies.alignment_paths["cv"],
            blank_idx1=self.dependencies.model_hyperparameters.blank_idx, name1="ground-truth",
            vocab_path1=self.dependencies.vocab_path, hdf_align_path2=cv_search_aligns,
            blank_idx2=self.dependencies.model_hyperparameters.blank_idx, name2="search",
            vocab_path2=self.dependencies.vocab_path, seq_tag=seq_tag)


class ReturnnFrameWiseSimpleBeamSearchPipeline(SearchPipeline):
  def get_decoding_experiment(self, corpus_key: str, dump_best_traces: bool) -> DecodingExperiment:
    returnn_config = get_segmental_recog_config(
      self.dependencies, self.variant_params, "dev", dump_output=dump_best_traces, length_scale=1.0)
    return SegmentalReturnnDecodingExperiment(
      dependencies=self.dependencies,
      returnn_config=returnn_config,
      variant_params=self.variant_params,
      variant_name=self.variant_name,
      checkpoint=self.checkpoint,
      epoch=self.epoch,
      dump_best_traces=dump_best_traces,
      corpus_key=corpus_key)


class RasrFrameWiseSegmentalBeamSearchPipeline(SearchPipeline):
  def __init__(
          self,
          length_norm: bool,
          label_pruning: Optional[float],
          label_pruning_limit: int,
          word_end_pruning: Optional[float],
          word_end_pruning_limit: int,
          full_sum_decoding: bool,
          allow_recombination: bool,
          max_segment_len: int,
          **kwargs
  ):
    super().__init__(**kwargs)
    self.length_norm = length_norm
    self.label_pruning = label_pruning
    self.label_pruning_limit = label_pruning_limit
    self.word_end_pruning = word_end_pruning
    self.word_end_pruning_limit = word_end_pruning_limit
    self.full_sum_decoding = full_sum_decoding
    self.allow_recombination = allow_recombination
    self.max_segment_len = max_segment_len

  def get_decoding_experiment(self, corpus_key: str, dump_best_traces: bool) -> DecodingExperiment:
    returnn_config = get_segmental_compile_config(self.variant_params, length_scale=1.0)
    return RasrDecodingExperiment(
      dependencies=self.dependencies,
      returnn_config=returnn_config,
      variant_params=self.variant_params,
      variant_name=self.variant_name,
      checkpoint=self.checkpoint,
      epoch=self.epoch,
      length_norm=self.length_norm,
      label_pruning=self.label_pruning,
      label_pruning_limit=self.label_pruning_limit,
      word_end_pruning=self.word_end_pruning,
      word_end_pruning_limit=self.word_end_pruning_limit,
      full_sum_decoding=self.full_sum_decoding,
      allow_recombination=self.allow_recombination,
      max_segment_len=self.max_segment_len,
      corpus_key=corpus_key,
      dump_best_traces=dump_best_traces
    )
