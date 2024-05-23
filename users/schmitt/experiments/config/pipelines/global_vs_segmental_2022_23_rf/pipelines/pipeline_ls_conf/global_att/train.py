from typing import Tuple, Optional, List
import itertools

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import GlobalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import _returnn_v2_train_step, from_scratch_training


def train_from_scratch(
        alias: str,
        config_builder: GlobalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 168,
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_scratch/%d-epochs_w-ctc-loss" % (n_epochs,)

    train_exp = GlobalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": time_rqmt
      },
      train_opts={
        "dataset_opts": {
          "use_speed_pert": True,
          "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}
        },
        "import_model_train_epoch1": None,
        "lr_opts": {"type": "dyn_lr_lin_warmup_invsqrt_decay"},
        "cleanup_old_models": {
          "keep_best_n": 4,
          "keep_last_n": 1,
          "keep": [n_epochs]
        },
        "train_step_func": _returnn_v2_train_step,
        "train_def": from_scratch_training,
      }
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()
    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_loss_ce",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }

    yield alias, checkpoint


def train_import_global_tf(
        alias: str,
        config_builder: GlobalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        const_lr_list: Tuple[float, ...],
        time_rqmt: int = 168,
):
  for n_epochs, const_lr in itertools.product(n_epochs_list, const_lr_list):
    train_alias = alias + f"/train_from_global_att_tf_checkpoint/standard-training/{n_epochs}-epochs_{const_lr}-const-lr_wo-ctc-loss"

    train_exp = GlobalTrainExperiment(
      config_builder=config_builder,
      alias=train_alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": time_rqmt
      },
      train_opts={
        "preload_from_files": {
          "pretrained_global_att_params": {
            "filename": external_checkpoints[default_import_model_name],
            "init_for_train": True,
          }
        },
        "train_def": from_scratch_training,
        "train_step_func": _returnn_v2_train_step,
        "batching": "random",
        "aux_loss_layers": None,
        "lr_opts": {
          "type": "const_then_linear",
          "const_lr": const_lr,
          "const_frac": 1 / 3,
          "final_lr": 1e-6,
          "num_epochs": n_epochs
        },
      }
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_loss_ce",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield train_alias, checkpoint
