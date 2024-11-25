from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LIBRISPEECH_CORPUS
from i6_core.text.label.subword_nmt.train import ReturnnTrainBpeJob
from i6_core.tools.git import CloneGitRepositoryJob

from typing import Dict
from abc import ABC, abstractmethod

from sisyphus import *


class LibrispeechBPE10025(LabelDefinition, ABC):
  def __init__(self):
    super().__init__()
    self.num_bpes = 10025

  @property
  def vocab_path(self) -> Path:
    return Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.vocab")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/u/zeineldeen/setups/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.vTq56NZ8STWt/output/bpe.codes")


class LibrispeechBPE1056(LabelDefinition, ABC):
  def __init__(self):
    super().__init__()
    self.num_bpes = 1056

  @property
  def vocab_path(self) -> Path:
    return Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab")

  @property
  def bpe_codes_path(self) -> Path:
    return Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.codes")


class LibrispeechBPE5048(LabelDefinition, ABC):
  def __init__(self):
    super().__init__()

    self.num_bpes = 5048

    subword_nmt_repo = CloneGitRepositoryJob(
      url="https://github.com/rwth-i6/subword-nmt",
      commit="6ba4515d684393496502b79188be13af9cad66e2",
      checkout_folder_name="subword-nmt",
      branch=None
    )
    self.train_bpe_job = ReturnnTrainBpeJob(
      text_file=LIBRISPEECH_CORPUS.corpus_to_txt_job.out_txt,
      bpe_size=5000,
      unk_label="<unk>",
      subword_nmt_repo=subword_nmt_repo.out_repository
    )

  @property
  def vocab_path(self) -> Path:
    return self.train_bpe_job.out_bpe_vocab

  @property
  def bpe_codes_path(self) -> Path:
    return self.train_bpe_job.out_bpe_codes
