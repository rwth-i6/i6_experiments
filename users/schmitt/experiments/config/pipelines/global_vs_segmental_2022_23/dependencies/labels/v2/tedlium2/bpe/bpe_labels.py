from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.bpe.bpe import LibrispeechBPE10025
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import GlobalModelHyperparameters

from typing import Dict

from sisyphus import *


class LibrispeechBPE10025Labels(LibrispeechBPE10025, LibrispeechLabelDefinition, GlobalLabelDefinition):
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
      sos_idx=0, target_num_labels=10025, sil_idx=None)
