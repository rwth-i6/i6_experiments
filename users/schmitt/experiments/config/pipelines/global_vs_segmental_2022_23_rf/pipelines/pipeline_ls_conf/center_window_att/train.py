from typing import Tuple, Optional, List, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import SegmentalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import _returnn_v2_train_step, from_scratch_training
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import (
  external_checkpoints,
  default_import_model_name,
  get_center_window_baseline_v1_tf_checkpoint
)


def train_center_window_att_from_scratch(
        alias: str,
        config_builder: SegmentalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 168,
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_scratch/%d-epochs_w-ctc-loss" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": time_rqmt
      },
      train_opts={
        "dataset_opts": {
          "use_speed_pert": False,
          "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}
        },
        "import_model_train_epoch1": None,
        "lr_opts": {"type": "dyn_lr_lin_warmup_invsqrt_decay"},
        "train_def": from_scratch_training,
        "train_step_func": _returnn_v2_train_step,
      }
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_label_model/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_center_window_att_import_global_tf(
        alias: str,
        config_builder: SegmentalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 168,
        train_opts: Optional[Dict] = None,
):
  _train_opts = {
    "preload_from_files": {
      "pretrained_global_att_params": {
        "filename": external_checkpoints[default_import_model_name],
        "init_for_train": True,
        "ignore_missing": True,  # because of length model params
      }
    },
    "train_def": from_scratch_training,
    "train_step_func": _returnn_v2_train_step,
    "batching": "random",
  }
  if train_opts:
    _train_opts.update(train_opts)

  for n_epochs in n_epochs_list:
    alias += "/train_from_global_att_tf_checkpoint/standard-training/%d-epochs_wo-ctc-loss" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": time_rqmt
      },
      train_opts=_train_opts
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_label_model/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_center_window_att_import_center_window_tf(
        alias: str,
        config_builder: SegmentalAttConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 168,
):
  for n_epochs in n_epochs_list:
    alias += "/train_from_center_window_baseline_v1_tf_checkpoint/standard-training/%d-epochs_wo-ctc-loss" % (n_epochs,)

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt={
        "time": time_rqmt
      },
      train_opts={
        # "preload_from_files": {
        #   "pretrained_global_att_params": {
        #     "filename": external_checkpoints["center-window_baseline-v1_tf"],
        #     "ignore_params_prefixes": ["emit_prob.", "s_length_model.", "target_embed_length_model."]
        #   }
        # },
        "import_model_train_epoch1": get_center_window_baseline_v1_tf_checkpoint(),
        "train_def": from_scratch_training,
        "train_step_func": _returnn_v2_train_step,
        "batching": "random",
        "aux_loss_layers": None,
      }
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_score_label_model/output_prob",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint
