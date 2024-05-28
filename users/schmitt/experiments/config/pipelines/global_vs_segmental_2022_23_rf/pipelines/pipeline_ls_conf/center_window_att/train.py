from typing import Tuple, Optional, List, Dict, Union, Callable

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import _returnn_v2_train_step, viterbi_training
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import (
  external_checkpoints,
  default_import_model_name,
)
from i6_experiments.users.schmitt.custom_load_params import load_missing_params


def train_center_window_att_viterbi_from_scratch(
        alias: str,
        config_builder: SegmentalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 80,
):
  batch_size = 15_000
  for n_epochs in n_epochs_list:
    alias += "/train_from_scratch/%d-epochs_w-ctc-loss" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": time_rqmt,
        "horovod_num_processes": 4,
        "distributed_launch_cmd": "torchrun"
      },
      train_opts={
        "dataset_opts": {
          "use_speed_pert": False,
          "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}
        },
        # "import_model_train_epoch1": None,
        "accum_grad_multiple_step": 4,
        "torch_distributed": {},
        "pos_emb_dropout": 0.1,
        "rf_att_dropout_broadcast": False,
        "batch_size": batch_size,
        "batching": "laplace:.1000",
        "lr_opts": {
          "type": "dyn_lr_piecewise_linear",
          "batch_size": batch_size,
          "num_epochs": n_epochs,
          "learning_rate": 1e-3,
        },
        "aux_loss_layers": None,
        "specaugment_steps": (5_000, 15_000, 25_000),
        "grad_scaler": None,
        "gradient_clip_global_norm": 5.0,
        "optimizer": {
          "class": "adamw",
          "weight_decay_modules_blacklist": [
            "rf.Embedding",
            "rf.LearnedRelativePositionalEncoding",
          ],
          "epsilon": 1e-16,
          "weight_decay": 1e-6,
        },
        "train_def": viterbi_training,
        "train_step_func": _returnn_v2_train_step,
      }
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_loss_non_blank_ce",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_center_window_att_viterbi_import_global_tf(
        alias: str,
        config_builder: SegmentalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        const_lr_list: Tuple[float, ...] = (1e-4,),
        time_rqmt: int = 80,
        alignment_augmentation_opts: Optional[Dict] = None,
):
  if not config_builder.use_att_ctx_in_state:
    # only randomly init FF weights, since only the input dim of the lstm layer is different
    custom_missing_load_func = load_missing_params
  else:
    custom_missing_load_func = None

  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      train_alias = alias + f"/train_from_global_att_tf_checkpoint/standard-training/{n_epochs}-epochs_{const_lr}-const-lr_wo-ctc-loss"
      if alignment_augmentation_opts:
        opts = alignment_augmentation_opts
        train_alias += f"_align-aug-{opts['num_iterations']}-iters_{opts['max_shift']}-max-shift"

      train_opts = {
        "preload_from_files": {
          "pretrained_global_att_params": {
            "filename": external_checkpoints[default_import_model_name],
            "init_for_train": True,
            "ignore_missing": True,  # because of length model params
          }
        },
        "aux_loss_layers": None,
        "accum_grad_multiple_step": 2,
        "optimizer": {"class": "adam", "epsilon": 1e-8},
        "train_def": viterbi_training,
        "train_step_func": _returnn_v2_train_step,
        "batching": "random",
        "lr_opts": {
          "type": "const_then_linear",
          "const_lr": const_lr,
          "const_frac": 1 / 3,
          "final_lr": 1e-6,
          "num_epochs": n_epochs
        },
        "alignment_augmentation_opts": alignment_augmentation_opts
      }
      if custom_missing_load_func:
        train_opts["preload_from_files"]["pretrained_global_att_params"]["custom_missing_load_func"] = custom_missing_load_func

      train_exp = SegmentalTrainExperiment(
        config_builder=config_builder,
        alias=train_alias,
        num_epochs=n_epochs,
        train_rqmt={
          "time": time_rqmt
        },
        train_opts=train_opts
      )
      checkpoints, model_dir, learning_rates = train_exp.run_train()

      checkpoint = {
        "model_dir": model_dir,
        "learning_rates": learning_rates,
        "key": "dev_loss_non_blank_ce",
        "checkpoints": checkpoints,
        "n_epochs": n_epochs
      }
      yield train_alias, checkpoint
