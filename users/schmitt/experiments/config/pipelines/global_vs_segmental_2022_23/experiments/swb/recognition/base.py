from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import LabelDefinition

# experiments
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.recognition import DecodingExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.search_errors import SearchErrorExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.analysis import AttentionWeightPlotter


from i6_core.returnn.training import Checkpoint
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.experiments.swb import default_tags_for_analysis

from sisyphus import *

from typing import Dict, Optional, List, Tuple, Type
from abc import ABC, abstractmethod


class SWBSearchPipeline(ABC):
  """
  Pipeline consists of decoding on the DEV set and then optional calculation of search errors, visualization of
  attention weights and visualization of alignment comparison on the CV set.
  """
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          base_alias: str,
          checkpoint: Checkpoint,
          corpus_key: str,
          calc_search_errors: bool = True,
          search_error_corpus_key: Optional[str] = "cv",
          plot_att_weights: bool = True,
          tags_for_analysis: Optional[List[str]] = None):

    self.tags_for_analysis = tags_for_analysis if tags_for_analysis is not None else default_tags_for_analysis
    self.plot_att_weights = plot_att_weights
    self.corpus_key = corpus_key
    self.calc_search_errors = calc_search_errors
    self.search_error_corpus_key = search_error_corpus_key
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.base_alias = base_alias
    self.checkpoint = checkpoint

  @abstractmethod
  def get_decoding_experiment(self, corpus_key, dump_best_traces) -> DecodingExperiment:
    pass

  @property
  @abstractmethod
  def alias(self) -> str:
    pass

  @property
  def analysis_sequences(self) -> List[Tuple[str, Path]]:
    return [("search", self.decode(corpus_key="cv", dump_best_traces=True))]

  @property
  @abstractmethod
  def att_weight_plotter_cls(self) -> Type[AttentionWeightPlotter]:
    pass

  @property
  @abstractmethod
  def att_weight_plotter_kwargs(self) -> Dict:
    pass

  @abstractmethod
  def calc_search_errors_func(self, search_aligns: Path):
    pass

  def decode(self, corpus_key: str, dump_best_traces: bool):
    decoding_exp = self.get_decoding_experiment(corpus_key=corpus_key, dump_best_traces=dump_best_traces)
    decoding_exp.run_eval()
    return decoding_exp.best_traces

  def get_cv_search_seqs(self):
    return self.decode(corpus_key="cv", dump_best_traces=True)

  def run(self):
    self.decode(corpus_key=self.corpus_key, dump_best_traces=False)

    if self.calc_search_errors:
      self.calc_search_errors_func(
        search_aligns=self.decode(corpus_key=self.search_error_corpus_key, dump_best_traces=True))

    if self.plot_att_weights:
      for hdf_alias, hdf_path in self.analysis_sequences:
        for seq_tag in self.tags_for_analysis:
          self.att_weight_plotter_cls(
            dependencies=self.dependencies,
            checkpoint=self.checkpoint,
            corpus_key="cv",
            seq_tag=seq_tag,
            hdf_target_path=self.decode(corpus_key="cv", dump_best_traces=True),
            hdf_alias=hdf_alias,
            variant_params=self.variant_params,
            base_alias=self.alias,
            **self.att_weight_plotter_kwargs).run()
