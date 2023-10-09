from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.formats import \
  RasrFormats
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import \
  SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.bpe.bpe import \
  TedLium2BPE1058
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.general import \
  TedLium2LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.hyperparameters import \
  SegmentalModelHyperparameters
from i6_experiments.users.schmitt.rasr.convert import BPEJSONVocabToRasrFormatsJob

from typing import Dict
import copy

from sisyphus import *


class TedLium2Bpe1058CtcAlignment(TedLium2BPE1058, TedLium2LabelDefinition, SegmentalLabelDefinition):
  """
  """

  @property
  def alias(self) -> str:
    return "bpe"

  @property
  def model_hyperparameters(self) -> SegmentalModelHyperparameters:
    return SegmentalModelHyperparameters(
      sos_idx=0, target_num_labels=1058, sil_idx=None, blank_idx=1057)

  @property
  def rasr_format_paths(self) -> RasrFormats:
    raise NotImplementedError

  @property
  def alignment_paths(self):
    return {
      "train": Path("/u/zeineldeen/setups/ubuntu_22_setups/2023-06-14--streaming-conf/work/i6_core/returnn/forward/ReturnnForwardJob.MkqI27U1sxyH/output/alignments-train.hdf", cached=True)
    }
