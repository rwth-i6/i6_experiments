from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import ModelHyperparameters, SegmentalModelHyperparameters, GlobalModelHyperparameters

from sisyphus import *

from abc import ABC, abstractmethod
from typing import Dict


class LabelDefinition(ABC):
  @property
  @abstractmethod
  def alias(self) -> str:
    pass

  @property
  @abstractmethod
  def stm_paths(self) -> str:
    pass

  @property
  @abstractmethod
  def stm_jobs(self) -> str:
    pass

  @property
  @abstractmethod
  def corpus_keys(self) -> str:
    pass

  @property
  @abstractmethod
  def oggzip_paths(self) -> str:
    pass

  @property
  @abstractmethod
  def segment_paths(self) -> str:
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
  @abstractmethod
  def model_hyperparameters(self) -> ModelHyperparameters:
    pass


class SegmentalLabelDefinition(LabelDefinition):
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
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    pass
