from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.librispeech import LibrispeechCorpora

from abc import ABC
from typing import Dict

from sisyphus import Path


LIBRISPEECH_CORPUS = LibrispeechCorpora()


class LibrispeechLabelDefinition(LabelDefinition, ABC):
  @property
  def stm_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.stm_paths

  @property
  def stm_jobs(self) -> str:
    return LIBRISPEECH_CORPUS.stm_jobs

  @property
  def corpus_keys(self) -> str:
    return LIBRISPEECH_CORPUS.corpus_keys

  @property
  def oggzip_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.oggzip_paths

  @property
  def segment_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.segment_paths

  @property
  def corpus_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.corpus_paths

  @property
  def arpa_lm_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.arpa_lm_paths

  @property
  def lm_word_list_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.lm_word_list_paths

  @property
  def nn_lm_meta_graph_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.nn_lm_meta_graph_paths

  @property
  def nn_lm_vocab_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.nn_lm_vocab_paths

  @property
  def nn_lm_checkpoint_paths(self) -> Dict[str, Path]:
    return LIBRISPEECH_CORPUS.nn_lm_checkpoint_paths
