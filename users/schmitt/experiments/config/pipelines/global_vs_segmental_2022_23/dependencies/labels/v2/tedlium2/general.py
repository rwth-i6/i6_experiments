from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.tedlium2 import TedLium2Corpora

from abc import ABC


TEDLIUM2_CORPUS = TedLium2Corpora()


class TedLium2LabelDefinition(LabelDefinition, ABC):
  @property
  def stm_paths(self) -> str:
    return TEDLIUM2_CORPUS.stm_paths

  @property
  def stm_jobs(self) -> str:
    return TEDLIUM2_CORPUS.stm_jobs

  @property
  def corpus_keys(self) -> str:
    return TEDLIUM2_CORPUS.corpus_keys

  @property
  def oggzip_paths(self) -> str:
    return TEDLIUM2_CORPUS.oggzip_paths

  @property
  def segment_paths(self) -> str:
    return TEDLIUM2_CORPUS.segment_paths
