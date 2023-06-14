from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.bpe.bpe import SWBBPE534
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.swb.general import SWBLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import GlobalModelHyperparameters

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
