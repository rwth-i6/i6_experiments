from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBOggZipCorpora

from abc import ABC
from typing import Tuple, Dict


class SWBLabelDefinition(LabelDefinition, ABC):
  @property
  def stm_paths(self) -> Dict:
    return SWBOggZipCorpora.stm_paths

  @property
  def stm_jobs(self) -> Dict:
    return SWBOggZipCorpora.stm_jobs

  @property
  def corpus_keys(self) -> Tuple:
    return SWBOggZipCorpora.corpus_keys

  @property
  def oggzip_paths(self) -> Dict:
    return SWBOggZipCorpora.oggzip_paths

  @property
  def segment_paths(self) -> Dict:
    return SWBOggZipCorpora.segment_paths
