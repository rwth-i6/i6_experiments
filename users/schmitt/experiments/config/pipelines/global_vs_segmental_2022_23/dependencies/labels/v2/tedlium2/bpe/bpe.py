from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class TedLium2BPE1058(LabelDefinition, ABC):
  @property
  def vocab_path(self) -> Path:
    pass

  @property
  def bpe_codes_path(self) -> Path:
    pass
