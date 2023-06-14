from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_ROOT

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.general import SegmentalLabelDefinition, GlobalLabelDefinition, LabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.segmental import get_train_config as get_segmental_train_config
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config_builder.config_builder import GlobalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.swb.returnn.config.global_ import get_train_config as get_global_train_config

from i6_core.returnn.training import ReturnnTrainingJob, Checkpoint
from i6_core.returnn.config import ReturnnConfig

from sisyphus import *

import copy
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple


class TrainExperiment(ABC):
  def __init__(
          self,
          dependencies: LabelDefinition,
          num_epochs: List[int],
          variant_params: Dict,
          base_alias: str,
          import_model_train_epoch1: Optional[Checkpoint] = None,
          initial_lr: Optional[float] = None
  ):
    self.dependencies = dependencies
    self.variant_params = variant_params
    self.num_epochs = num_epochs
    self.import_model_train_epoch1 = import_model_train_epoch1
    self.returnn_python_exe = self.variant_params["returnn_python_exe"]
    self.returnn_root = self.variant_params["returnn_root"]
    self.initial_lr = initial_lr

    self.alias = "%s/train" % base_alias

  @property
  @abstractmethod
  def returnn_config(self):
    pass

  def run_training(self) -> Tuple[Dict[int, Checkpoint], Path]:
    train_job = ReturnnTrainingJob(
      copy.deepcopy(self.returnn_config),
      num_epochs=self.num_epochs[-1],
      keep_epochs=self.num_epochs,
      log_verbosity=5,
      returnn_python_exe=self.returnn_python_exe,
      returnn_root=self.returnn_root,
      mem_rqmt=24,
      time_rqmt=30)
    train_job.add_alias(self.alias)
    alias = train_job.get_one_alias()
    tk.register_output(alias + "/config", train_job.out_returnn_config_file)
    tk.register_output(alias + "/models", train_job.out_model_dir)
    tk.register_output(alias + "/learning_rates", train_job.out_learning_rates)
    tk.register_output(alias + "/plot_se", train_job.out_plot_se)
    tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)

    return train_job.out_checkpoints, train_job.out_learning_rates


class SegmentalTrainExperiment(TrainExperiment):
  def __init__(
          self,
          dependencies: SegmentalLabelDefinition,
          cv_alignment: Optional[Path] = None,
          train_alignment: Optional[Path] = None,
          length_scale: float = 1.0,
          **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies
    self.length_scale = length_scale

    assert (
            cv_alignment is None and train_alignment is None) or (
            cv_alignment is not None and train_alignment is not None)
    self.alignments = {
      "train": train_alignment,
      "devtrain": train_alignment,
      "cv": cv_alignment} if cv_alignment is not None else None

  @property
  def returnn_config(self):
    return get_segmental_train_config(
      dependencies=self.dependencies,
      alignments=self.alignments,
      variant_params=self.variant_params,
      import_model_train_epoch1=self.import_model_train_epoch1,
      length_scale=self.length_scale,
      load=None,
      initial_lr=self.initial_lr
    )


class GlobalTrainExperiment(TrainExperiment):
  def __init__(self, dependencies: GlobalLabelDefinition, **kwargs):
    super().__init__(dependencies=dependencies, **kwargs)

    self.dependencies = dependencies

  @property
  def returnn_config(self):
    if self.variant_params["config"]["enc_type"] == "conf-mohammad-11.4":
      return GlobalConfigBuilder(
        dependencies=self.dependencies,
        variant_params=self.variant_params
      ).get_train_config()
    else:
      return get_global_train_config(
        self.dependencies,
        self.variant_params,
        load=None,
        import_model_train_epoch1=self.import_model_train_epoch1,
        initial_lr=self.initial_lr
      )
