from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.recognition import SegmentalReturnnDecodingExperiment, RasrDecodingExperiment, DecodingExperiment, GlobalReturnnDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.search_errors import SegmentalSearchErrorExperiment, GlobalSearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import AlignmentComparer, SegmentalAttentionWeightsPlotter, GlobalAttentionWeightsPlotter
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_recog_config as get_global_recog_config

from i6_core.returnn.training import Checkpoint

from sisyphus import *

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.base import SWBSearchPipeline

from typing import Dict, Optional, List, Tuple, Type
from abc import ABC, abstractmethod


class ReturnnLabelSyncBeamSearchPipeline(SWBSearchPipeline):
  def __init__(
          self,
          dependencies: GlobalLabelDefinition,
          beam_size: int,
          **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)
    self.dependencies = dependencies
    self.beam_size = beam_size

  @property
  def alias(self) -> str:
    return "%s/returnn_label_wise_beam_search" % self.base_alias

  @property
  def analysis_sequences(self) -> List[Tuple[str, Path]]:
    # return [("ground_truth", self.dependencies.label_paths["cv"])] + super().analysis_sequences
    return super().analysis_sequences
  def get_decoding_experiment(self, corpus_key, dump_best_traces) -> DecodingExperiment:
    returnn_config = get_global_recog_config(
      self.dependencies, self.variant_params, corpus_key,
      dump_output=dump_best_traces, beam_size=self.beam_size)
    return GlobalReturnnDecodingExperiment(
      dependencies=self.dependencies,
      returnn_config=returnn_config,
      variant_params=self.variant_params,
      checkpoint=self.checkpoint,
      dump_best_traces=dump_best_traces,
      corpus_key=corpus_key,
      base_alias=self.alias)

  @property
  def att_weight_plotter_cls(self) -> Type[GlobalAttentionWeightsPlotter]:
    return GlobalAttentionWeightsPlotter

  @property
  def att_weight_plotter_kwargs(self) -> Dict:
    return {}

  def calc_search_errors_func(self, search_aligns: Path):
    GlobalSearchErrorExperiment(
      dependencies=self.dependencies,
      checkpoint=self.checkpoint,
      variant_params=self.variant_params,
      search_targets=search_aligns,
      ref_targets=self.dependencies.label_paths[self.search_error_corpus_key],
      corpus_key=self.search_error_corpus_key,
      base_alias=self.alias).create_calc_search_error_job()


def run_returnn_label_sync_decoding(
        dependencies: GlobalLabelDefinition,
        variant_params: Dict,
        base_alias: str,
        checkpoint: Checkpoint,
        test_corpora_keys: List[str],
        calc_search_errors: bool,
        search_error_corpus_key: Optional[str],
        plot_att_weights: bool = True,
        beam_size: int = 12):
  assert not calc_search_errors or search_error_corpus_key is not None, "`search_error_corpus_key` needs to be set"

  # Simple RETURNN frame-wise beam search
  for corpus_key in test_corpora_keys:
    ReturnnLabelSyncBeamSearchPipeline(
      dependencies=dependencies,
      variant_params=variant_params,
      base_alias=base_alias,
      checkpoint=checkpoint,
      corpus_key=corpus_key,
      calc_search_errors=calc_search_errors,
      search_error_corpus_key=search_error_corpus_key,
      plot_att_weights=plot_att_weights,
      beam_size=beam_size
    ).run()

