from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import ModelHyperparameters, SegmentalModelHyperparameters, GlobalModelHyperparameters

from sisyphus import *

from abc import ABC, abstractmethod
from typing import Dict, Tuple


class LabelDefinition(ABC):
  def __init__(self):
    self.hdf_targets = {}
  @property
  @abstractmethod
  def alias(self) -> str:
    pass

  @property
  @abstractmethod
  def stm_paths(self) -> Dict:
    pass

  @property
  @abstractmethod
  def stm_jobs(self) -> Dict:
    pass

  @property
  @abstractmethod
  def corpus_keys(self) -> Tuple:
    pass

  @property
  @abstractmethod
  def oggzip_paths(self) -> Dict:
    pass

  @property
  @abstractmethod
  def segment_paths(self) -> Dict:
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

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    return {}


class GlobalLabelDefinition(LabelDefinition):
  @property
  @abstractmethod
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    pass

  @property
  def label_paths(self) -> Dict[str, Path]:
    return {}
