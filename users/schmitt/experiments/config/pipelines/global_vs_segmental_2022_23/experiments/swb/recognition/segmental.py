from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.recognition import SegmentalReturnnDecodingExperiment, RasrDecodingExperiment, DecodingExperiment, GlobalReturnnDecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.realignment import RasrRealignmentExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.search_errors import SegmentalSearchErrorExperiment, GlobalSearchErrorExperiment, SearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.search_errors import SegmentalSearchErrorExperiment, GlobalSearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import AlignmentComparer, SegmentalAttentionWeightsPlotter, GlobalAttentionWeightsPlotter
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.realignment import run_rasr_segmental_realignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_recog_config as get_segmental_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_recog_config as get_global_recog_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_compile_config as get_segmental_compile_config

from i6_core.returnn.training import Checkpoint

from sisyphus import *

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb.recognition.base import SWBSearchPipeline

from typing import Dict, Optional, List, Tuple, Type
from abc import ABC, abstractmethod


class SWBSegmentalSearchPipeline(SWBSearchPipeline, ABC):
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          compare_alignments: bool,
          length_scale: float,
          cv_realignment: Optional[Path],
          **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies
    self.length_scale = length_scale
    self.compare_alignments = compare_alignments
    self.cv_realignment = cv_realignment

    self.length_norm = False
    self.max_segment_len = 20

  @property
  def att_weight_plotter_cls(self) -> Type[SegmentalAttentionWeightsPlotter]:
    return SegmentalAttentionWeightsPlotter

  @property
  def att_weight_plotter_kwargs(self) -> Dict:
    return {"length_scale": self.length_scale}

  @property
  def analysis_sequences(self) -> List[Tuple[str, Path]]:
    # return [("ground_truth", self.dependencies.alignment_paths["cv"])] + super().analysis_sequences
    if self.cv_realignment is None:
      return super().analysis_sequences
    else:
      return [("realignment", self.cv_realignment)] + super().analysis_sequences

  def calc_search_errors_func(self, search_aligns: Path):
    SegmentalSearchErrorExperiment(
      dependencies=self.dependencies,
      checkpoint=self.checkpoint,
      variant_params=self.variant_params,
      search_targets=search_aligns,
      ref_targets=self.cv_realignment if self.cv_realignment is not None else self.dependencies.alignment_paths["cv"],
      corpus_key=self.search_error_corpus_key,
      length_scale=self.length_scale,
      base_alias=self.alias).create_calc_search_error_job()

  def run(self):
    super().run()

    if self.compare_alignments:
      for i, (hdf_alias1, hdf1) in enumerate(self.analysis_sequences[:-1]):
        for j, (hdf_alias2, hdf2) in enumerate(self.analysis_sequences[i+1:]):
          AlignmentComparer(
            hdf_align_path1=hdf1,
            blank_idx1=self.dependencies.model_hyperparameters.blank_idx,
            name1=hdf_alias1,
            vocab_path1=self.dependencies.vocab_path,
            hdf_align_path2=hdf2,
            blank_idx2=self.dependencies.model_hyperparameters.blank_idx,
            name2=hdf_alias2,
            vocab_path2=self.dependencies.vocab_path,
            seq_tags=self.tags_for_analysis,
            corpus_key="cv",
            base_alias=self.alias).run()


class ReturnnFrameWiseSimpleBeamSearchPipeline(SWBSegmentalSearchPipeline):
  def __init__(
          self,
          beam_size: int,
          use_recomb: bool,
          **kwargs):
    super().__init__(**kwargs)

    self.beam_size = beam_size
    self.use_recomb = use_recomb

  @property
  def alias(self) -> str:
    return "%s/returnn_frame_wise_beam_search" % self.base_alias

  def get_decoding_experiment(self, corpus_key, dump_best_traces) -> DecodingExperiment:
    returnn_config = get_segmental_recog_config(
      self.dependencies,
      self.variant_params,
      corpus_key,
      dump_output=dump_best_traces,
      length_scale=self.length_scale,
      beam_size=self.beam_size,
      use_recomb=self.use_recomb)
    return SegmentalReturnnDecodingExperiment(
      dependencies=self.dependencies,
      returnn_config=returnn_config,
      variant_params=self.variant_params,
      checkpoint=self.checkpoint,
      dump_best_traces=dump_best_traces,
      corpus_key=corpus_key,
      base_alias=self.alias)


class RasrFrameWiseSegmentalBeamSearchPipeline(SWBSegmentalSearchPipeline):
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
          time_rqmt: int,
          mem_rqmt: int,
          concurrent: int,
          **kwargs):
    super().__init__(**kwargs)

    self.length_norm = length_norm
    self.label_pruning = label_pruning
    self.label_pruning_limit = label_pruning_limit
    self.word_end_pruning = word_end_pruning
    self.word_end_pruning_limit = word_end_pruning_limit
    self.full_sum_decoding = full_sum_decoding
    self.allow_recombination = allow_recombination
    self.max_segment_len = max_segment_len
    self.concurrent = concurrent

    self.time_rqmt = time_rqmt
    self.mem_rqmt = mem_rqmt

  @property
  def alias(self) -> str:
    return "%s/rasr_frame_wise_segmental_beam_search" % self.base_alias

  def get_decoding_experiment(self, corpus_key, dump_best_traces) -> DecodingExperiment:
    returnn_config = get_segmental_compile_config(self.variant_params, length_scale=self.length_scale)
    return RasrDecodingExperiment(
      dependencies=self.dependencies,
      returnn_config=returnn_config,
      variant_params=self.variant_params,
      checkpoint=self.checkpoint,
      length_norm=self.length_norm,
      label_pruning=self.label_pruning,
      label_pruning_limit=self.label_pruning_limit,
      word_end_pruning=self.word_end_pruning,
      word_end_pruning_limit=self.word_end_pruning_limit,
      full_sum_decoding=self.full_sum_decoding,
      allow_recombination=self.allow_recombination,
      max_segment_len=self.max_segment_len,
      corpus_key=corpus_key,
      dump_best_traces=dump_best_traces,
      concurrent=self.concurrent,
      time_rqmt=self.time_rqmt,
      mem_rqmt=self.mem_rqmt,
      base_alias=self.alias)


def run_returnn_simple_segmental_decoding(
        dependencies: SegmentalLabelDefinition,
        variant_params: Dict,
        base_alias: str,
        checkpoint: Checkpoint,
        test_corpora_keys: List[str],
        calc_search_errors: bool,
        search_error_corpus_key: Optional[str],
        cv_realignment: Optional[Path],
        compare_alignments: bool = True,
        plot_att_weights: bool = True,
        beam_size: int = 12,
        use_recomb: bool = False,
        length_scale: float = 1.0):
  assert not calc_search_errors or search_error_corpus_key is not None, "`search_error_corpus_key` needs to be set"

  # Simple RETURNN frame-wise beam search on DEV
  for corpus_key in test_corpora_keys:
    ReturnnFrameWiseSimpleBeamSearchPipeline(
      length_scale=length_scale,
      dependencies=dependencies,
      variant_params=variant_params,
      base_alias=base_alias,
      checkpoint=checkpoint,
      corpus_key=corpus_key,
      calc_search_errors=calc_search_errors,
      search_error_corpus_key=search_error_corpus_key,
      compare_alignments=compare_alignments,
      plot_att_weights=plot_att_weights,
      beam_size=beam_size,
      use_recomb=use_recomb,
      cv_realignment=cv_realignment
    ).run()


def run_rasr_segmental_decoding(
        dependencies: SegmentalLabelDefinition,
        variant_params: Dict,
        base_alias: str,
        checkpoint: Checkpoint,
        test_corpora_keys: List[str],
        calc_search_errors: bool,
        cv_realignment: Optional[Path],
        search_error_corpus_key: Optional[str] = None,
        length_norm: bool = False,
        label_pruning: float = None,
        label_pruning_limit: int = 12,
        word_end_pruning: float = None,
        word_end_pruning_limit: int = 12,
        full_sum_decoding: bool = False,
        allow_recombination: bool = True,
        max_segment_len: int = 20,
        length_scale: float = 1.0,
        compare_alignments: bool = True,
        plot_att_weights: bool = True,
        time_rqmt: Optional[int] = None,
        mem_rqmt: Optional[int] = None,
        concurrent: int = 1
):
  assert not calc_search_errors or search_error_corpus_key is not None, "`search_error_corpus_key` needs to be set"

  if time_rqmt is None:
    time_rqmt = 20 // concurrent
  if mem_rqmt is None:
    mem_rqmt = 20 // concurrent

  for corpus_key in test_corpora_keys:
    RasrFrameWiseSegmentalBeamSearchPipeline(
      length_norm=length_norm,
      label_pruning=label_pruning,
      label_pruning_limit=label_pruning_limit,
      word_end_pruning=word_end_pruning,
      word_end_pruning_limit=word_end_pruning_limit,
      full_sum_decoding=full_sum_decoding,
      allow_recombination=allow_recombination,
      max_segment_len=max_segment_len,
      dependencies=dependencies,
      variant_params=variant_params,
      base_alias=base_alias,
      checkpoint=checkpoint,
      length_scale=length_scale,
      corpus_key=corpus_key,
      calc_search_errors=calc_search_errors,
      search_error_corpus_key=search_error_corpus_key,
      compare_alignments=compare_alignments,
      plot_att_weights=plot_att_weights,
      concurrent=concurrent,
      time_rqmt=time_rqmt,
      mem_rqmt=mem_rqmt,
      cv_realignment=cv_realignment).run()
