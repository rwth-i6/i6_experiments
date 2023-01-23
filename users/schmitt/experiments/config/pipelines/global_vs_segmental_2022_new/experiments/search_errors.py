from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import LabelDefinition, SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.general.rasr.exes import RasrExecutables
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.returnn.config.seg import get_train_config as get_segmental_train_config

from i6_private.users.schmitt.returnn.tools import CalcSearchErrorJob

from i6_core.returnn.training import Checkpoint

from abc import ABC, abstractmethod
from typing import Dict

from sisyphus import *


class SearchErrorExperiment(ABC):
  def __init__(
          self, dependencies: LabelDefinition, variant_params: Dict, variant_name: str,
          checkpoint: Checkpoint, search_targets: Path, epoch: int
    ):
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.checkpoint = checkpoint
    self.search_targets = search_targets
    self.alias = "%s/returnn_no-recomb_length-scale-1.0_beam-12/epoch_%d/search_errors" % (variant_name, epoch)

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
      rasr_config=self.dependencies.rasr_config_paths["feature_extraction"],
      rasr_nn_trainer_exe=RasrExecutables.nn_trainer_path,
      segment_file=self.dependencies.segment_paths["cv"],
      blank_idx=self.blank_idx,
      model_type=self.model_type,
      label_name=self.label_name,
      search_targets=self.search_targets,
      ref_targets=self.dependencies.alignment_paths["cv"],
      max_seg_len=-1, length_norm=False)
    calc_search_err_job.add_alias(self.alias)
    tk.register_output(self.alias, calc_search_err_job.out_search_errors)


class SegmentalSearchErrorExperiment(SearchErrorExperiment):
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

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
    return get_segmental_train_config(self.dependencies, self.variant_params, load=self.checkpoint, length_scale=1.0)


class GlobalSearchErrorExperiment(SearchErrorExperiment):
  def __init__(self, dependencies: LabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

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
      pass




