from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class Phonemes(LabelDefinition, ABC):
  @property
  @abstractmethod
  def vocab_path(self) -> Path:
    pass
