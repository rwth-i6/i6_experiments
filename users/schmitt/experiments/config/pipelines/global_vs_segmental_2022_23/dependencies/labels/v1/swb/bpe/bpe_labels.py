from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe import BPE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.bpe.bpe_rna_alignment import RNABPE
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v1.swb.general import GlobalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import GlobalModelHyperparameters

from i6_experiments.users.schmitt.alignment.alignment import DumpNonBlanksFromAlignmentJob

from typing import Dict

from sisyphus import *


class BPELabels(BPE, GlobalLabelDefinition):
  """
  These are the BPE labels of the SWB corpus.
  """
  def __init__(self):
    super().__init__()

    self.rna_ref = RNABPE()

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def train_segment_paths(self) -> Dict[str, Path]:
    return LabelDefinition.get_default_train_segment_paths()

  @property
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    return GlobalModelHyperparameters(
      sos_idx=0, target_num_labels=1030, sil_idx=None)

  @property
  def label_paths(self) -> Dict[str, Path]:
    label_paths = {
      corpus_key: DumpNonBlanksFromAlignmentJob(
        alignment=alignment, blank_idx=self.rna_ref.model_hyperparameters.blank_idx, time_rqmt=1).out_labels
      for corpus_key, alignment in self.rna_ref.alignment_paths.items()}
    label_paths["cv300"] = label_paths["cv"]
    label_paths["cv_test"] = label_paths["cv"]

    return label_paths

  @property
  def vocab_path(self) -> Path:
    return Path('/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k')

  @property
  def vocab_dict(self) -> Dict[str, Path]:
    return {
      "vocab_file": self.vocab_path,
      "bpe_file": self.bpe_codes_path,
      "seq_postfix": [self.model_hyperparameters.sos_idx]}
