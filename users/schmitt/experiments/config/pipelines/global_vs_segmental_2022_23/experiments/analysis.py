from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import LabelDefinition, SegmentalLabelDefinition, GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.graph import ReturnnGraph
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables

from i6_experiments.users.schmitt.alignment.alignment import CompareAlignmentsJob
from i6_experiments.users.schmitt.returnn.tools import DumpAttentionWeightsJob
from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_train_config as get_segmental_train_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_train_config as get_global_train_config

from i6_core.returnn.training import Checkpoint
from i6_core.returnn.config import ReturnnConfig

from sisyphus import *

from abc import ABC, abstractmethod
from typing import Dict, Optional, List


class AttentionWeightPlotter(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          checkpoint: Checkpoint,
          corpus_key: str,
          seq_tag: str,
          hdf_target_path: Path,
          hdf_alias: str,
          variant_params: Dict,
          base_alias: str
  ):
    self.checkpoint = checkpoint
    self.variant_params = variant_params
    self.hdf_target_path = hdf_target_path
    self.seq_tag = seq_tag
    self.corpus_key = corpus_key
    self.dependencies = dependencies

    self.alias = "%s/analysis/%s/att_weights/%s/%s" % (base_alias, corpus_key, seq_tag.replace("/", "_"), hdf_alias)

  @property
  @abstractmethod
  def returnn_config(self) -> ReturnnConfig:
    pass

  @property
  @abstractmethod
  def blank_idx(self) -> int:
    pass

  @property
  @abstractmethod
  def model_type(self) -> str:
    pass

  @property
  @abstractmethod
  def label_name(self) -> str:
    pass

  def run(self):
    att_weights_data_path = DumpAttentionWeightsJob(
      returnn_config=self.returnn_config,
      model_type=self.model_type,
      rasr_config=self.dependencies.rasr_config_paths["feature_extraction"][self.corpus_key],
      blank_idx=self.blank_idx,
      label_name=self.label_name,
      rasr_nn_trainer_exe=RasrExecutables.nn_trainer_path,
      hdf_targets=self.hdf_target_path,
      seq_tag=self.seq_tag).out_data

    plot_weights_job = PlotAttentionWeightsJob(
      data_path=att_weights_data_path,
      blank_idx=self.blank_idx,
      json_vocab_path=self.dependencies.vocab_path,
      time_red=6,
      seq_tag=self.seq_tag)
    plot_weights_job.add_alias(self.alias)
    tk.register_output(plot_weights_job.get_one_alias(), plot_weights_job.out_plot)


class SegmentalAttentionWeightsPlotter(AttentionWeightPlotter):
  def __init__(self, dependencies: SegmentalLabelDefinition, length_scale: float, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies
    self.length_scale = length_scale

  @property
  def blank_idx(self) -> int:
    return self.dependencies.model_hyperparameters.blank_idx

  @property
  def model_type(self) -> str:
    return "seg"

  @property
  def label_name(self) -> str:
    return "alignment"

  @property
  def returnn_config(self) -> ReturnnConfig:
    return get_segmental_train_config(
      dependencies=self.dependencies,
      alignments=None,
      variant_params=self.variant_params,
      load=self.checkpoint,
      length_scale=self.length_scale)


class GlobalAttentionWeightsPlotter(AttentionWeightPlotter):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  @property
  def blank_idx(self) -> Optional[int]:
    return None

  @property
  def model_type(self) -> str:
    return "glob"

  @property
  def label_name(self) -> str:
    return "bpe"

  @property
  def returnn_config(self) -> ReturnnConfig:
    return get_global_train_config(self.dependencies, self.variant_params, load=self.checkpoint)


class AlignmentComparer:
  def __init__(
          self,
          hdf_align_path1: Path,
          blank_idx1: int,
          name1: str,
          vocab_path1: Path,
          hdf_align_path2: Path,
          blank_idx2: int,
          name2: str,
          vocab_path2: Path,
          seq_tags: List[str],
          corpus_key: str,
          base_alias: str
  ):

    self.hdf_align_path1 = hdf_align_path1
    self.blank_idx1 = blank_idx1
    self.name1 = name1
    self.vocab_path1 = vocab_path1
    self.hdf_align_path2 = hdf_align_path2
    self.blank_idx2 = blank_idx2
    self.name2 = name2
    self.vocab_path2 = vocab_path2
    self.seq_tags = seq_tags

    self.alias = "%s/analysis/%s/alignment_compare/%s-%s" % (base_alias, corpus_key, self.name1, self.name2)

  def run(self):
    compare_aligns_job = CompareAlignmentsJob(
      hdf_align1=self.hdf_align_path1,
      hdf_align2=self.hdf_align_path2,
      seq_tags=self.seq_tags,
      blank_idx1=self.blank_idx1,
      blank_idx2=self.blank_idx2,
      vocab1=self.vocab_path1,
      vocab2=self.vocab_path2,
      name1=self.name1,
      name2=self.name2)
    compare_aligns_job.add_alias(self.alias)
    tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)
