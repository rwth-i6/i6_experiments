from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.bpe.bpe import SWBBPE534, SWBBPE1030
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.general import SWBLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.bpe.bpe_alignments import SwbBpe1030RnaAlignment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import GlobalModelHyperparameters
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_ROOT, RETURNN_EXE_NEW

from i6_experiments.users.schmitt.alignment.alignment import DumpNonBlanksFromAlignmentJob

from typing import Dict

from sisyphus import *


class SWBBPE534Labels(SWBBPE534, SWBLabelDefinition, GlobalLabelDefinition):
  """
  These are the BPE labels of the SWB corpus.
  """
  def __init__(self):
    super().__init__()

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    return GlobalModelHyperparameters(
      sos_idx=0, target_num_labels=534, sil_idx=None)


class SWBBPE1030Labels(SWBBPE1030, SWBLabelDefinition, GlobalLabelDefinition):
  """
  These are the BPE labels of the SWB corpus.
  """
  def __init__(self):
    super().__init__()

    self.rna_ref = SwbBpe1030RnaAlignment()
    self.hdf_targets = {
      corpus_key: DumpNonBlanksFromAlignmentJob(
        alignment=alignment,
        blank_idx=self.rna_ref.model_hyperparameters.blank_idx,
        time_rqmt=1,
        returnn_root=RETURNN_ROOT,
        returnn_python_exe=RETURNN_EXE_NEW,
      ).out_labels
      for corpus_key, alignment in self.rna_ref.hdf_targets.items()}

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    return GlobalModelHyperparameters(
      sos_idx=0, target_num_labels=1030, sil_idx=None, target_num_labels_wo_blank=1030)

  @property
  def segment_paths(self) -> Dict:
    return self.rna_ref.segment_paths
