from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.sentencepiece.sentencepiece import LibrispeechSP10240
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.general import LibrispeechLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import SegmentalModelHyperparameters
from sisyphus import Path

from abc import ABC
from typing import Dict


class LibrispeechSP10240Alignment(LibrispeechSP10240, LibrispeechLabelDefinition, SegmentalLabelDefinition, ABC):
  def __init__(self):
    super().__init__()

    self._alignment_paths = None

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  @property
  def alias(self) -> str:
    return "att-transducer-alignment"

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    if self._alignment_paths is None:
      raise ValueError("Alignments first need to be set externally!")
    return self._alignment_paths

  @alignment_paths.setter
  def alignment_paths(self, value):
    assert isinstance(value, dict)
    assert self._alignment_paths is None, "Alignment paths are already set!"
    assert "train" in value and "cv" in value and "devtrain" in value
    self._alignment_paths = value


class LibrispeechSP10240AlignmentJointModel(LibrispeechSP10240Alignment):
  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=10240, sil_idx=None, blank_idx=0, target_num_labels_wo_blank=10240)


class LibrispeechSP10240AlignmentSepModel(LibrispeechSP10240Alignment):
  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=10241, sil_idx=None, blank_idx=10240, target_num_labels_wo_blank=10240)
