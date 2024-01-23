from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class SWBBPE534(LabelDefinition, ABC):
  @property
  def vocab_path(self) -> Path:
    return Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.FLNETa4J87YO/output/bpe.vocab")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.FLNETa4J87YO/output/bpe.codes")


class SWBBPE1030(LabelDefinition, ABC):
  @property
  def vocab_path(self) -> Path:
    return Path("/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k")
