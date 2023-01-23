from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from typing import Dict
from abc import ABC

from sisyphus import *


class BPE(LabelDefinition, ABC):
  @property
  def segment_paths(self) -> Dict[str, Path]:
    return {
      "train": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train", cached=True),
      "devtrain": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_train_head3000", cached=True),
      "cv": Path("/u/schmitt/experiments/transducer/config/dependencies/seg_cv_head3000", cached=True),
      "dev": None, "hub5e_01": None, "rt03s": None}

  @property
  def vocab_path(self) -> Path:
    return Path('/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k')

  @property
  def bpe_codes_path(self) -> Path:
    return Path('/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k')
