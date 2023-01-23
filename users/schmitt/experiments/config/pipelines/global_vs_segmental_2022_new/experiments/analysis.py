from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import LabelDefinition, SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.returnn.graph import ReturnnGraph
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.config import RasrConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.exes import RasrExecutables

from i6_private.users.schmitt.returnn.tools import DumpForwardJob, CompileTFGraphJob, RASRDecodingJob, \
  CombineAttentionPlotsJob, DumpPhonemeAlignJob, AugmentBPEAlignmentJob, FindSegmentsToSkipJob, ModifySeqFileJob, \
  ConvertCTMBPEToWordsJob, RASRLatticeToCTMJob, CompareAlignmentsJob, DumpAttentionWeightsJob, PlotAttentionWeightsJob, \
  DumpNonBlanksFromAlignmentJob, CalcSearchErrorJob, RemoveLabelFromAlignmentJob, WordsToCTMJob, SwitchLabelInAlignmentJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.returnn.config.seg import get_train_config as get_segmental_train_config

from i6_core.returnn.training import Checkpoint

from sisyphus import *

from abc import ABC, abstractmethod
from typing import Dict


class AttentionWeightPlotter(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          checkpoint: Checkpoint,
          corpus_key: str,
          seq_tag: str,
          hdf_target_path: Path,
          hdf_alias: str,
          variant_name: str,
          variant_params: Dict
  ):
    self.checkpoint = checkpoint
    self.variant_params = variant_params
    self.hdf_target_path = hdf_target_path
    self.seq_tag = seq_tag
    self.corpus_key = corpus_key
    self.dependencies = dependencies

    self.alias = "%s/analysis/att_weights/%s/%s" % (variant_name, seq_tag.replace("/", "_"), hdf_alias)

  @property
  @abstractmethod
  def returnn_config(self):
    pass

  @property
  @abstractmethod
  def blank_idx(self):
    pass

  def plot_att_weights(self):
    att_weights_data_path = DumpAttentionWeightsJob(
      returnn_config=self.returnn_config,
      model_type="seg",
      rasr_config=self.dependencies.rasr_config_paths["feature-extraction"][self.corpus_key],
      blank_idx=self.blank_idx,
      label_name="alignment",
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
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  @property
  def blank_idx(self):
    return self.dependencies.model_hyperparameters.blank_idx

  @property
  def returnn_config(self):
    return get_segmental_train_config(self.dependencies, self.variant_params, load=self.checkpoint, length_scale=1.0)


class AlignmentComparer:
  def __init__(
          self,
          variant_name,
          hdf_align_path1: Path,
          blank_idx1: int,
          name1: str,
          vocab_path1: Path,
          hdf_align_path2: Path,
          blank_idx2: int,
          name2: str,
          vocab_path2: Path,
          seq_tag: str
  ):
    alias = "%s/analysis/align_compare/%s/%s-%s" % (variant_name, seq_tag.replace("/", "_"), name1, name2)

    compare_aligns_job = CompareAlignmentsJob(
      hdf_align1=hdf_align_path1,
      hdf_align2=hdf_align_path2,
      seq_tag=seq_tag,
      blank_idx1=blank_idx1,
      blank_idx2=blank_idx2,
      vocab1=vocab_path1,
      vocab2=vocab_path2,
      name1=name1,
      name2=name2)
    compare_aligns_job.add_alias(alias)
    tk.register_output(compare_aligns_job.get_one_alias(), compare_aligns_job.out_align)
