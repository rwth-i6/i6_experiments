from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.corpora.corpora import SWBCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from abc import ABC, abstractmethod
from typing import Dict

from sisyphus import *


class SegmentalLabelDefinition(LabelDefinition):
  @property
  @abstractmethod
  def alignment_paths(self) -> Dict[str, Path]:
    pass

  @property
  @abstractmethod
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    pass

