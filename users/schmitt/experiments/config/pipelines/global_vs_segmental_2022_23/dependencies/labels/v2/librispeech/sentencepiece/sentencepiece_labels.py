from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.sentencepiece.sentencepiece import LibrispeechSP10240
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import GlobalModelHyperparameters
from sisyphus import Path


class LibrispeechSP10240Labels(LibrispeechSP10240, LibrispeechLabelDefinition, GlobalLabelDefinition):
  def __init__(self):
    super().__init__()

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> GlobalModelHyperparameters:
    return GlobalModelHyperparameters(
      sos_idx=0, target_num_labels=10240, sil_idx=None, target_num_labels_wo_blank=10240)
