from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class LibrispeechBPE10025(LabelDefinition, ABC):
  @property
  def vocab_path(self) -> Path:
    return Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes")


class LibrispeechBPE1056(LabelDefinition, ABC):
  @property
  def vocab_path(self) -> Path:
    return Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes")
