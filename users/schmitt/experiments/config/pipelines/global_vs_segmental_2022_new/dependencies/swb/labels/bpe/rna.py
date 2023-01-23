from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.bpe.bpe import BPE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.segmental import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.formats import RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.hyperparameters import SegmentalModelHyperparameters

from typing import Dict

from sisyphus import *


class RNABPE(BPE, SegmentalLabelDefinition):
  @property
  def alias(self) -> str:
    return "rna-bpe"

  @staticmethod
  def get_alignment_paths() -> Dict[str, Path]:
    return {
      "train": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
        cached=True),
      "devtrain": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-train.hdf",
        cached=True),
      "cv": Path(
        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/merboldt_swb_transducer/rna-tf2.blank0.enc6l-grow2l.scratch-lm.rdrop02.lm1-1024.attwb5-drop02.l2_1e_4.mlr50.epoch-150.swap.data-dev.hdf",
        cached=True)}

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    return RNABPE.get_alignment_paths()

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, blank_idx=1030, target_num_labels=1031, sil_idx=None)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    return RasrFormats(
      vocab_path=self.vocab_path,
      blank_idx=self.model_hyperparameters.blank_idx,
      corpus_alias=self.dataset_alias,
      label_alias=self.alias,
      lexicon_path=Path("/work/asr3/zeyer/schmitt/sisyphus_work_dirs/swb1/dependencies/bpe/irie/my_new_lex"),
      blank_allophone_state_idx=4119
    )
