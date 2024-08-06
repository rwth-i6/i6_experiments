from typing import Tuple, Optional, List
import itertools

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechGlobalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import _returnn_v2_train_step, from_scratch_training
from i6_experiments.users.schmitt.custom_load_params import load_missing_params


def train_from_scratch(
        alias: str,
        config_builder: LibrispeechGlobalAttConformerConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        batch_size: int = 15_000,
        time_rqmt: int = 168,
        use_mgpu: bool = True,
):
  for n_epochs in n_epochs_list:
    alias += (
            f"/train_from_scratch/{n_epochs}-epochs_wo-ctc-loss"
            f"{'_mgpu-4' if use_mgpu else ''}"
    )

    train_rqmt = {
      "time": time_rqmt,
    }

    train_opts = {
      "dataset_opts": {
        "use_speed_pert": True,
        "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}
      },
      "import_model_train_epoch1": None,
      "accum_grad_multiple_step": 4,
      "pos_emb_dropout": 0.1,
      "rf_att_dropout_broadcast": False,
      "batch_size": batch_size,
      "batching": "laplace:.1000",
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
      "lr_opts": {
        "type": "dyn_lr_piecewise_linear",
        "batch_size": batch_size,
        "num_epochs": n_epochs,
        "learning_rate": 1e-3,
      },
      "cleanup_old_models": {
        "keep_best_n": 4,
        "keep_last_n": 1,
        "keep": [n_epochs]
      },
      "train_step_func": _returnn_v2_train_step,
      "train_def": from_scratch_training,
    }

    if use_mgpu:
      train_rqmt.update({
        "horovod_num_processes": 4,
        "distributed_launch_cmd": "torchrun"
      })
      train_opts["torch_distributed"] = {"reduce_type": "param", "param_sync_step": 100}

    train_exp = GlobalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt=train_rqmt,
      train_opts=train_opts,
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
        config_builder: LibrispeechGlobalAttConformerConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        const_lr_list: Tuple[float, ...],
        time_rqmt: int = 80,
        import_model_name: str = default_import_model_name,
):
  if not config_builder.use_att_ctx_in_state:
    # only randomly init FF weights, since only the input dim of the lstm layer is different
    custom_missing_load_func = load_missing_params
  else:
    custom_missing_load_func = None

  for n_epochs, const_lr in itertools.product(n_epochs_list, const_lr_list):
    train_alias = alias + f"/train_from_{import_model_name}/standard-training/{n_epochs}-epochs_{const_lr}-const-lr_wo-ctc-loss"

    train_opts = {
      "preload_from_files": {
        "pretrained_global_att_params": {
          "filename": external_checkpoints[import_model_name],
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
    if custom_missing_load_func:
      train_opts["preload_from_files"]["pretrained_global_att_params"]["custom_missing_load_func"] = custom_missing_load_func
    if config_builder.label_decoder_state != "nb-lstm":
      train_opts["preload_from_files"]["pretrained_global_att_params"]["ignore_missing"] = True

    train_exp = GlobalTrainExperiment(
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
      "key": "dev_loss_ce",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield train_alias, checkpoint
