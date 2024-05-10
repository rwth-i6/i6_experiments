from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import \
  SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.bpe.bpe import \
  TedLium2BPE1057
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.general import \
  TedLium2LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import \
  SegmentalModelHyperparameters
from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob

from typing import Dict
import copy

from sisyphus import *


class TedLium2Bpe1057CtcAlignment(TedLium2BPE1057, TedLium2LabelDefinition, SegmentalLabelDefinition):
  """
  """
  def __init__(self):
    super().__init__()

    self._alignment_paths = None

  @property
  def alias(self) -> str:
    return "global-att-aux-ctc-alignments"

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=1058, sil_idx=None, blank_idx=1057, target_num_labels_wo_blank=1057)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  # @property
  # def alignment_paths(self):
  #   return {
  #     "train": Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-06-14--streaming-conf/work/i6_core/returnn/forward/ReturnnForwardJob.MkqI27U1sxyH/output/alignments-train.hdf", cached=True)
  #   }

  @property
  def alignment_paths(self) -> Dict[str, Path]:
    if self._alignment_paths is None:
      raise ValueError("Alignments first need to be set externally!")
    return self._alignment_paths

  @alignment_paths.setter
  def alignment_paths(self, value):
    assert isinstance(value, dict)
    assert self._alignment_paths is None, "Alignment paths are already set!"
    self._alignment_paths = value
