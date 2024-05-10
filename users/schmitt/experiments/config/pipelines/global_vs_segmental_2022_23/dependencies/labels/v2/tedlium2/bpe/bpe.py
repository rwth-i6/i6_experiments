from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class TedLium2BPE1057(LabelDefinition, ABC):
  @property
  def vocab_path(self) -> Path:
    return Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-06-14--streaming-conf/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.vocab")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-04-17--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.Jc3xHSQQbXD9/output/bpe.codes")
