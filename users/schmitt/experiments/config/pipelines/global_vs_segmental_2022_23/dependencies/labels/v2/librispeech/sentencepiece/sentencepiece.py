from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LIBRISPEECH_CORPUS
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.tools.git import CloneGitRepositoryJob

from typing import Dict, Optional
from abc import ABC, abstractmethod

from sisyphus import *


class LibrispeechSP10240(LabelDefinition, ABC):
  @property
  def model_path(self) -> Path:
    return Path("/u/zeyer/setups/combined/2021-05-31/work/i6_core/text/label/sentencepiece/train/TrainSentencePieceJob.ofYcs4cMRS8T/output/spm_out.model")

  @property
  def vocab_path(self) -> Optional[Path]:
    return Path("/work/asr4/zeyer/setups-data/combined/2021-05-31/work/i6_core/text/label/sentencepiece/vocab/ExtractSentencePieceVocabJob.mSRAzk018au3/output/spm.vocab")

  @property
  def bpe_codes_path(self) -> Optional[Path]:
    return None

