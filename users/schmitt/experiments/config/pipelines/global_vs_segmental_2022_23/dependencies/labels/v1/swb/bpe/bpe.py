from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.general import LabelDefinition

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class BPE(LabelDefinition, ABC):
  @property
  @abstractmethod
  def vocab_path(self) -> Path:
    pass

  @property
  def bpe_codes_path(self) -> Path:
    return Path('/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k')
