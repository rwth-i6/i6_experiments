from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.labels.general import SegmentalLabelDefinition
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_new.dependencies.swb.returnn.config.seg import get_train_config as get_segmental_train_config

from i6_core.returnn.training import ReturnnTrainingJob

from sisyphus import *

import copy
from abc import ABC, abstractmethod
from typing import Dict, List


class TrainExperiment(ABC):
  def __init__(
          self,
          num_epochs: List[int],
          variant_name: str,
          variant_params: Dict,
          alias: str
  ):
    self.variant_params = variant_params
    self.variant_name = variant_name
    self.num_epochs = num_epochs

    self.alias = "%s/%s" % (variant_name, alias)

  @property
  @abstractmethod
  def returnn_config(self):
    pass

  def run_training(self) -> Dict[int, Path]:
    train_job = ReturnnTrainingJob(
      copy.deepcopy(self.returnn_config),
      num_epochs=self.num_epochs[-1],
      keep_epochs=self.num_epochs,
      log_verbosity=5,
      returnn_python_exe="/u/rossenbach/bin/returnn_tf2.3_launcher.sh",
      returnn_root="/u/schmitt/src/returnn",
      mem_rqmt=24,
      time_rqmt=30)
    train_job.add_alias(self.alias)
    alias = train_job.get_one_alias()
    tk.register_output(alias + "/config", train_job.out_returnn_config_file)
    tk.register_output(alias + "/models", train_job.out_model_dir)
    tk.register_output(alias + "/learning_rates", train_job.out_learning_rates)
    tk.register_output(alias + "/plot_se", train_job.out_plot_se)
    tk.register_output(alias + "/plot_lr", train_job.out_plot_lr)

    return train_job.out_checkpoints


class SegmentalTrainExperiment(TrainExperiment):
  def __init__(self, dependencies: SegmentalLabelDefinition, **kwargs):
    super().__init__(**kwargs)

    self.dependencies = dependencies

  @property
  def returnn_config(self):
    return get_segmental_train_config(self.dependencies, self.variant_params, load=None, length_scale=1.0)
