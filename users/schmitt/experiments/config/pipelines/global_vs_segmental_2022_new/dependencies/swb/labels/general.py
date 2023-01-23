from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb import dataset_alias as swb_dataset_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.corpora.corpora import SWBCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.hyperparameters import ModelHyperparameters, SegmentalModelHyperparameters, GlobalModelHyperparameters

from i6_core.corpus.convert import CorpusToStmJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from sisyphus import *

from abc import ABC, abstractmethod
from typing import Dict


class LabelDefinition(ABC):
  def __init__(self):
    self.dataset_alias = swb_dataset_alias

    self.feature_cache_paths = SWBCorpora.feature_cache_paths
    self.corpus_paths = SWBCorpora.corpus_paths
    self.test_stm_paths = SWBCorpora.test_stm_paths
    self.corpus_keys = SWBCorpora.corpus_keys
    self.corpus_mapping = SWBCorpora.corpus_mapping

  @property
  @abstractmethod
  def alias(self) -> str:
    pass

  @property
  @abstractmethod
  def segment_paths(self) -> Dict[str, Path]:
    pass

  @property
  @abstractmethod
  def vocab_path(self) -> Path:
    pass

  @property
  @abstractmethod
  def bpe_codes_path(self) -> Path:
    pass

  @property
  def vocab_dict(self) -> Dict[str, Path]:
    return {
      "bpe_file": self.bpe_codes_path,
      "vocab_file": self.vocab_path
    }

  @property
  @abstractmethod
  def model_hyperparameters(self) -> ModelHyperparameters:
    pass

  @property
  def rasr_config_paths(self) -> Dict[str, Dict[str, Path]]:
    return {
      "feature_extraction": {corpus_key: RasrConfigBuilder.get_feature_extraction_config(
        self.segment_paths[corpus_key],
        self.feature_cache_paths[SWBCorpora.corpus_mapping[corpus_key]],
        self.corpus_paths[SWBCorpora.corpus_mapping[corpus_key]]) for corpus_key in self.corpus_keys}}

  @property
  def stm_paths(self) -> Dict[str, Path]:
    return dict(
      cv=CorpusToStmJob(
        bliss_corpus=FilterCorpusBySegmentsJob(
          bliss_corpus=self.corpus_paths["train"], segment_file=self.segment_paths["cv"]
        ).out_corpus).out_stm_path,
      **self.test_stm_paths
    )


class SegmentalLabelDefinition(LabelDefinition):
  @property
  @abstractmethod
  def alignment_paths(self) -> Dict[str, Path]:
    pass

  @property
  @abstractmethod
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    pass

  @property
  @abstractmethod
  def rasr_format_paths(self) -> RasrFormats:
    pass


class GlobalLabelDefinition(LabelDefinition):
  @property
  @abstractmethod
  def label_paths(self) -> Dict[str, Path]:
    pass

  @property
  @abstractmethod
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    pass
