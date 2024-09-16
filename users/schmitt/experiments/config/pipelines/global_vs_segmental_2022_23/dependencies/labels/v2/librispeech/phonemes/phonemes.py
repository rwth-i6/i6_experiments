from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LIBRISPEECH_CORPUS
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.tools.git import CloneGitRepositoryJob

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class LibrispeechPhonemes41(LabelDefinition, ABC):
  def __init__(self):
    super().__init__()

    self._vocab_path = None

  @property
  def vocab_path(self) -> Path:
    assert self._vocab_path is not None, "vocab path not set"
    return self._vocab_path

  @vocab_path.setter
  def vocab_path(self, value: Path):
    self._vocab_path = value


class LibrispeechWords(LabelDefinition, ABC):
  def __init__(self):
    super().__init__()

    self._vocab_path = None

  @property
  def vocab_path(self) -> Path:
    assert self._vocab_path is not None, "vocab path not set"
    return self._vocab_path

  @vocab_path.setter
  def vocab_path(self, value: Path):
    self._vocab_path = value
