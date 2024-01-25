from sisyphus import tk, Path
import copy
from typing import Dict, List, Optional, Tuple

from i6_core.returnn.training import AverageTFCheckpointsJob, GetBestEpochJob, Checkpoint, ReturnnTrainingJob, GetBestTFCheckpointJob

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.base import ConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.swb import SWBCorpus
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.search_errors import calc_search_errors
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.att_weights import dump_att_weights
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE, RETURNN_CURRENT_ROOT, RETURNN_ROOT


class TrainExperiment:
  def __init__(
          self,
          config_builder: ConfigBuilder,
          alias: str,
          n_epochs: int,
          import_model_train_epoch1: Optional[Checkpoint] = None,
          lr_opts: Optional[Dict] = None,
          cleanup_old_models: Optional[Dict] = None,
          no_ctc_loss: bool = False,
          train_mini_lstm_opts: Optional[Dict] = None,
  ):
    self.alias = alias
    self.import_model_train_epoch1 = import_model_train_epoch1
    self.config_builder = config_builder
    self.n_epochs = n_epochs
    self.lr_opts = lr_opts
    self.no_ctc_loss = no_ctc_loss
    self.train_mini_lstm_opts = train_mini_lstm_opts
    self.cleanup_old_models = cleanup_old_models if cleanup_old_models is not None else {
      "keep_best_n": 1, "keep_last_n": 1}

  def get_train_opts(self):
    return {
      "cleanup_old_models": self.cleanup_old_models,
      "lr_opts": self.lr_opts,
      "import_model_train_epoch1": self.import_model_train_epoch1,
      "no_ctc_loss": self.no_ctc_loss,
      "train_mini_lstm_opts": self.train_mini_lstm_opts,
    }

  def run_train(self) -> Tuple[Dict[int, Checkpoint], Path, Path]:
    checkpoints, model_dir, learning_rates = run_train(
      config_builder=self.config_builder,
      variant_params=self.config_builder.variant_params,
      n_epochs=self.n_epochs,
      train_opts=self.get_train_opts(),
      alias=self.alias
    )

    return checkpoints, model_dir, learning_rates

  @staticmethod
  def get_best_checkpoint(model_dir: Path, learning_rates: Path, key: str, index: int = 0) -> Checkpoint:
    return GetBestTFCheckpointJob(
      model_dir=model_dir, learning_rates=learning_rates, key=key, index=index
    ).out_checkpoint

  @staticmethod
  def get_averaged_checkpoint(
          model_dir: Path,
          learning_rates: Path,
          key: str,
          best_n: int,
          config_builder: ConfigBuilder,
  ) -> Checkpoint:
    best_epochs = []
    for i in range(best_n):
      best_epochs.append(GetBestEpochJob(
        model_dir=model_dir,
        learning_rates=learning_rates,
        key=key,
        index=i
      ).out_epoch)

    return AverageTFCheckpointsJob(
      model_dir=model_dir,
      epochs=best_epochs,
      returnn_python_exe=config_builder.variant_params["returnn_python_exe"],
      returnn_root=config_builder.variant_params["returnn_root"]
    ).out_checkpoint


class SegmentalTrainExperiment(TrainExperiment):
  def __init__(
          self,
          align_targets: Dict[str, Path],
          chunking_opts: Optional[Dict] = None,
          only_train_length_model: bool = False,
          align_augment: bool = False,
          **kwargs,
  ):
    super().__init__(**kwargs)

    self.align_augment = align_augment
    self.align_targets = align_targets
    self.chunking_opts = chunking_opts
    self.only_train_length_model = only_train_length_model

  def get_train_opts(self):
    train_opts = super().get_train_opts()

    train_opts.update({
      "dataset_opts": {"hdf_targets": self.align_targets},
      "only_train_length_model": self.only_train_length_model,
      "chunking": self.chunking_opts,
      "align_augment": self.align_augment,
    })

    return train_opts


class GlobalTrainExperiment(TrainExperiment):
  def get_train_opts(self):
    train_opts = super().get_train_opts()

    train_opts.update({
      "tf_session_opts": {"gpu_options": {"per_process_gpu_memory_fraction": 0.95}},
      "max_seq_length": {"targets": 75}
    })

    return train_opts


def run_train(
        config_builder: ConfigBuilder,
        variant_params: Dict,
        n_epochs: int,
        train_opts: Dict,
        alias: str,
  ):
  train_config_builder = copy.deepcopy(config_builder)
  train_config = train_config_builder.get_train_config(opts=train_opts)

  alias = alias + "/train"
  if train_opts.get("train_mini_lstm_opts") is not None:  # need to check for None because it can be {}
    alias = alias + "_mini_lstm"
    if train_opts["train_mini_lstm_opts"].get("use_eos", False):
      alias = alias + "_w_eos"
    else:
      alias = alias + "_wo_eos"

  train_job = ReturnnTrainingJob(
    train_config,
    num_epochs=n_epochs,
    keep_epochs=[n_epochs],
    log_verbosity=5,
    returnn_python_exe=variant_params["returnn_python_exe"],
    returnn_root=variant_params["returnn_root"],  # might need to change this to old returnn when using positional embedding option
    mem_rqmt=24,
    time_rqmt=30)
  train_job.add_alias(alias)
  tk.register_output(train_job.get_one_alias() + "/models", train_job.out_model_dir)
  tk.register_output(train_job.get_one_alias() + "/plot_lr", train_job.out_plot_lr)

  return train_job.out_checkpoints, train_job.out_model_dir, train_job.out_learning_rates
