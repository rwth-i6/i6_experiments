from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.labels.general import LabelDefinition, SegmentalLabelDefinition, GlobalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_train_config as get_segmental_train_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_train_config as get_global_train_config

from i6_experiments.users.schmitt.returnn.tools import CalcSearchErrorJob

from i6_core.returnn.training import Checkpoint

from abc import ABC, abstractmethod
from typing import Dict

from sisyphus import *


class SearchErrorExperiment(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          variant_params: Dict,
          checkpoint: Checkpoint,
          search_targets: Path,
          ref_targets: Path,
          corpus_key: str,
          base_alias: str
    ):
    self.ref_targets = ref_targets
    self.corpus_key = corpus_key
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.checkpoint = checkpoint
    self.search_targets = search_targets
    self.alias = "%s/search_errors_%s" % (base_alias, corpus_key)

  @property
  @abstractmethod
  def model_type(self):
    pass

  @property
  @abstractmethod
  def label_name(self):
    pass

  @property
  @abstractmethod
  def blank_idx(self):
    pass

  @property
  @abstractmethod
  def returnn_config(self):
    pass

  def create_calc_search_error_job(self):
    calc_search_err_job = CalcSearchErrorJob(
      returnn_config=self.returnn_config,
      rasr_config=self.dependencies.rasr_config_paths["feature_extraction"][self.corpus_key],
      rasr_nn_trainer_exe=RasrExecutables.nn_trainer_path,
      segment_file=self.dependencies.segment_paths[self.corpus_key],
      blank_idx=self.blank_idx,
      model_type=self.model_type,
      label_name=self.label_name,
      search_targets=self.search_targets,
      ref_targets=self.ref_targets,
      max_seg_len=-1, length_norm=False)
    calc_search_err_job.add_alias(self.alias)
    tk.register_output(self.alias, calc_search_err_job.out_search_errors)


class SegmentalSearchErrorExperiment(SearchErrorExperiment):
  def __init__(self, dependencies: SegmentalLabelDefinition, length_scale: float, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies
    self.length_scale = length_scale

  @property
  def model_type(self):
    return "seg"

  @property
  def label_name(self):
    return "alignment"

  @property
  def blank_idx(self):
    return self.dependencies.model_hyperparameters.blank_idx

  @property
  def returnn_config(self):
    return get_segmental_train_config(
      dependencies=self.dependencies,
      alignments=None,
      variant_params=self.variant_params,
      load=self.checkpoint,
      length_scale=self.length_scale)


class GlobalSearchErrorExperiment(SearchErrorExperiment):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  @property
  def model_type(self):
    return "glob"

  @property
  def label_name(self):
    return "bpe"

  @property
  def blank_idx(self):
    # here, we just return a random idx, as it is not used in the case of a global att model
    return 0

  @property
  def returnn_config(self):
      return get_global_train_config(self.dependencies, self.variant_params, load=self.checkpoint)
