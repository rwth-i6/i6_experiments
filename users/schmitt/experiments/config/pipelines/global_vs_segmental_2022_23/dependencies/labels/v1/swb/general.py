from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBSprintCorpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import ModelHyperparameters, SegmentalModelHyperparameters, GlobalModelHyperparameters
from i6_experiments.users.schmitt.util.util import ModifySeqFileJob

from i6_core.corpus.segments import ShuffleAndSplitSegmentsJob
from i6_core.corpus.convert import CorpusToStmJob
from i6_core.corpus.filter import FilterCorpusBySegmentsJob

from sisyphus import *

from abc import ABC, abstractmethod
from typing import Dict


class LabelDefinition(ABC):
  def __init__(self):
    self.feature_cache_paths = SWBSprintCorpora.feature_cache_paths
    self.test_stm_paths = SWBSprintCorpora.test_stm_paths
    self.corpus_keys = SWBSprintCorpora.corpus_keys
    self.corpus_paths = SWBSprintCorpora.corpus_paths

  @property
  @abstractmethod
  def alias(self) -> str:
    pass

  @staticmethod
  def get_default_train_segment_paths() -> Dict[str, Path]:
    return {
      "train": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train", cached=True),
      "devtrain": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train_head3000", cached=True),
      "cv": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000", cached=True)}

  @property
  @abstractmethod
  def train_segment_paths(self) -> Dict[str, Path]:
    pass

  @property
  def segment_paths(self) -> Dict[str, Path]:
    return dict(
      **self.train_segment_paths,
      cv300=ShuffleAndSplitSegmentsJob(
        segment_file=self.train_segment_paths["cv"], split={"10": 0.1, "90": 0.9}).out_segments["10"],
      **SWBSprintCorpora.segment_paths)

  @property
  def realignment_segment_paths(self) -> Dict[str, Path]:
    """
    This is just a quick fix for now. During realignment, RASR fails to do realignment on some sequences with the error
    message "no encoder features given?!". This results in some seqs missing in the final alignment cache and this
    leads to the ChooseBestAlignmentJob to fail because there are seqs missing in the realignment. Therefore,
    I here define segment files which exclude these sequences for which realignment fails. Depending on the label type,
    e.g. rna-bpe, I need to define different segments.
    :return:
    """
    return self.segment_paths

  @property
  @abstractmethod
  def vocab_path(self) -> Path:
    pass

  @property
  @abstractmethod
  def vocab_dict(self) -> Dict[str, Path]:
    pass

  @property
  @abstractmethod
  def model_hyperparameters(self) -> ModelHyperparameters:
    pass

  @property
  def stm_jobs(self) -> Dict[str, CorpusToStmJob]:
    stm_jobs = [CorpusToStmJob(
      bliss_corpus=FilterCorpusBySegmentsJob(
        bliss_corpus=SWBSprintCorpora.corpus_paths[corpus_key],
        segment_file=self.segment_paths[corpus_key]).out_corpus
    ) if corpus_key in ["cv", "cv300", "dev400"] else CorpusToStmJob(
      bliss_corpus=SWBSprintCorpora.corpus_paths[corpus_key]
    ) for corpus_key in SWBSprintCorpora.test_corpus_keys]

    return {corpus_key: stm_job for corpus_key, stm_job in zip(SWBSprintCorpora.test_corpus_keys, stm_jobs)}

  @property
  def stm_paths(self) -> Dict[str, Path]:
    return dict(
      cv=self.stm_jobs["cv"].out_stm_path,
      cv300=self.stm_jobs["cv300"].out_stm_path,
      **self.test_stm_paths)

  @property
  def rasr_config_paths(self) -> Dict[str, Dict[str, Path]]:
    return {"feature_extraction": {
      corpus_key: RasrConfigBuilder.get_feature_extraction_config(
        self.segment_paths[corpus_key],
        self.feature_cache_paths[corpus_key],
        self.corpus_paths[corpus_key]) for corpus_key in self.corpus_keys}}


class SegmentalLabelDefinition(LabelDefinition):
  @property
  @abstractmethod
  def alignment_paths(self) -> Dict[str, Path]:
    pass

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
  def label_paths(self) -> Dict[str, Path]:
    pass

  @property
  @abstractmethod
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    pass
