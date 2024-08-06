from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple, Union
from abc import ABC, abstractmethod

from i6_core.returnn.training import Checkpoint, ReturnnTrainingJob

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import SegmentalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import GlobalConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import GlobalAttConfigBuilderRF, SegmentalAttConfigBuilderRF, ConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.ctc import CtcConfigBuilder


class TrainExperiment(ABC):
  def __init__(
          self,
          config_builder: Union[ConfigBuilder, ConfigBuilderRF],
          alias: str,
          num_epochs: int,
          train_opts: Optional[Dict] = None,
          train_rqmt: Optional[Dict] = None
  ):
    self.alias = alias
    self.config_builder = config_builder
    self.num_epochs = num_epochs
    self.train_opts = self.default_train_opts
    if train_opts is not None:
      _train_opts = copy.deepcopy(train_opts)
      self.train_opts.update(_train_opts)
      if "cleanup_old_models" not in self.train_opts:
        self.train_opts["cleanup_old_models"] = {
          "keep_best_n": 4, "keep_last_n": 1, "keep": [num_epochs]
        }

    self.train_rqmt = train_rqmt if train_rqmt is not None else {}
    self.alias = self.alias + "/train"
    if self.train_opts.get("train_mini_lstm_opts") is not None:  # need to check for None because it can be {}
      self.alias += "_mini_lstm"

      if self.train_opts["train_mini_lstm_opts"].get("use_se_loss", False):
        self.alias += "/w_se_loss"
      else:
        self.alias += "/wo_se_loss"

      self.alias += "_%d_epochs" % num_epochs

  @property
  @abstractmethod
  def default_train_opts(self) -> Dict:
    pass

  def run_train(self) -> Tuple[Dict[int, Checkpoint], Path, Path]:
    config_builder = copy.deepcopy(self.config_builder)
    train_config = config_builder.get_train_config(opts=self.train_opts)

    train_job = ReturnnTrainingJob(
      train_config,
      num_epochs=self.num_epochs,
      log_verbosity=5,
      returnn_python_exe=config_builder.variant_params["returnn_python_exe"],
      returnn_root=config_builder.variant_params["returnn_root"],
      mem_rqmt=self.train_rqmt.get("mem", 24),
      time_rqmt=self.train_rqmt.get("time", 30),
      cpu_rqmt=self.train_rqmt.get("cpu", 4),
      horovod_num_processes=self.train_rqmt.get("horovod_num_processes", None),
      distributed_launch_cmd=self.train_rqmt.get("distributed_launch_cmd", "mpirun"),
    )
    if self.train_rqmt.get("gpu_mem", 11) > 11:
      train_job.rqmt["gpu_mem"] = self.train_rqmt["gpu_mem"]

    train_job.add_alias(self.alias)
    tk.register_output(train_job.get_one_alias() + "/models", train_job.out_model_dir)
    tk.register_output(train_job.get_one_alias() + "/plot_lr", train_job.out_plot_lr)

    return train_job.out_checkpoints, train_job.out_model_dir, train_job.out_learning_rates


class SegmentalTrainExperiment(TrainExperiment):
  def __init__(self, config_builder: Union[SegmentalConfigBuilder, SegmentalAttConfigBuilderRF], **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)

  @property
  def default_train_opts(self) -> Dict:
    return {
      "chunking_opts": None,
      # "import_model_train_epoch1": external_checkpoints[default_import_model_name],
      "lr_opts": {
        "type": "const_then_linear",
        "const_lr": 1e-4,
        "const_frac": 1 / 3,
        "final_lr": 1e-6,
        "num_epochs": self.num_epochs
      },
      "only_train_length_model": False,
      "no_ctc_loss": False,
      "train_mini_lstm_opts": None,
    }


class GlobalTrainExperiment(TrainExperiment):
  def __init__(self, config_builder: Union[GlobalConfigBuilder, GlobalAttConfigBuilderRF], **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)

  @property
  def default_train_opts(self) -> Dict:
    return {
      # "import_model_train_epoch1": external_checkpoints[default_import_model_name],
      "lr_opts": {
        "type": "const_then_linear",
        "const_lr": 1e-4,
        "const_frac": 1 / 3,
        "final_lr": 1e-6,
        "num_epochs": self.num_epochs
      },
      "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
      "max_seq_length": {"targets": 75},
      "train_mini_lstm_opts": None,
    }


class CtcTrainExperiment(TrainExperiment):
  def __init__(self, config_builder: CtcConfigBuilder, **kwargs):
    super().__init__(config_builder=config_builder, **kwargs)

  @property
  def default_train_opts(self) -> Dict:
    return {
      # "import_model_train_epoch1": external_checkpoints[default_import_model_name],
      # "dataset_opts": {"hdf_targets": {
      #   "train": LibrispeechBPE10025_LABELS_WITH_SILENCE._label_paths["train"],
      #   "devtrain": LibrispeechBPE10025_LABELS_WITH_SILENCE._label_paths["train"],
      #   "cv": LibrispeechBPE10025_LABELS_WITH_SILENCE._label_paths["train"]
      # }},
      "lr_opts": {
        "type": "const_then_linear",
        "const_lr": 1e-4,
        "const_frac": 1 / 3,
        "final_lr": 1e-6,
        "num_epochs": self.num_epochs
      },
      # "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
      # "max_seq_length": {"targets": 75},
    }
