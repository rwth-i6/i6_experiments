from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora

from abc import ABC


LIBRISPEECH_CORPUS = LibrispeechCorpora()


class LibrispeechLabelDefinition(LabelDefinition, ABC):
  @property
  def stm_paths(self) -> str:
    return LIBRISPEECH_CORPUS.stm_paths

  @property
  def stm_jobs(self) -> str:
    return LIBRISPEECH_CORPUS.stm_jobs

  @property
  def corpus_keys(self) -> str:
    return LIBRISPEECH_CORPUS.corpus_keys

  @property
  def oggzip_paths(self) -> str:
    return LIBRISPEECH_CORPUS.oggzip_paths

  @property
  def segment_paths(self) -> str:
    return LIBRISPEECH_CORPUS.segment_paths
