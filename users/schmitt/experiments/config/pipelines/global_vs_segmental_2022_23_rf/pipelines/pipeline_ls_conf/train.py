import copy
from typing import Tuple, Optional, List, Dict, Union, Callable

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechSegmentalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import (
  external_checkpoints,
)

from i6_core.returnn.training import PtCheckpoint


def _get_optimizer_alias(optimizer_opts: Dict):
  return (
    f"opt-{optimizer_opts['class']}-eps-{optimizer_opts['epsilon']}-wd-{optimizer_opts.get('weight_decay', 0.0)}"
  )


def _get_reduced_input_len(input_len: int, config_builder: LibrispeechSegmentalAttConformerConfigBuilderRF):
  return int(input_len - config_builder.red_subtrahend + config_builder.red_factor - 1) // config_builder.red_factor


regularization_opts = {
  "v1": {
    "weight_decay": 1e-6,
  },
  "v2": {
    "weight_dropout": 0.1,
    "weight_decay": 1e-2,
    "att_dropout": 0.2,
    "target_embed_dropout": 0.1,
  },
  "v3": {
    "weight_decay": 0.01,
  },
  "v4": {
    "weight_decay": 2e-2,
  },
  "v5": {
    "weight_decay": 4e-2,
  },
}


def get_common_train_opts_rqmt(
        n_epochs: int,
        training_type: str,
        time_rqmt: int = 80,
        batch_size: int = 15_000,
        use_mgpu: bool = True,
        ctc_aux_loss_layers: Optional[Tuple[int, ...]] = None,
        ctc_aux_loss_focal_loss_factors: Optional[Tuple[float, ...]] = None,
        ctc_aux_loss_scales: Optional[Tuple[float, ...]] = None,
        use_curriculum_learning: bool = True,
        lr_scheduling_opts: Optional[Dict] = None,
        regularization_type: str = "v1",
        use_speed_pert: bool = True,
        checkpoint_alias: Optional[str] = None,
        checkpoint_path: Optional[PtCheckpoint] = None,
        gpu_mem_rqmt: int = 11,
        keep_epochs: Optional[List[int]] = None,
        filter_data_len: Optional[float] = None,
        filter_target_len: Optional[float] = None,
        accum_grad_multiple_step: int = 4,
        cluster_reservation_string: Optional[str] = None,
        use_torch_amp: bool = False,
        random_seed: Optional[int] = None,
        disable_enc_self_att_until_epoch: Optional[int] = None,
        cutoff_initial_silence: bool = False,
        use_speed_pert_w_flip: bool = False,
):
  alias = (
    f"/{training_type}_from_{'scratch' if checkpoint_alias is None else checkpoint_alias}"
    f"/{n_epochs}-ep_bs-{batch_size}{'_mgpu-4' if use_mgpu else ''}_{'w' if use_speed_pert else 'wo'}-sp"
    f"{'-w-flip' if use_speed_pert_w_flip else ''}"
    f"_{'curric' if use_curriculum_learning else 'no-curric'}"
    f"_reg-{regularization_type}{'_filter-data-' + str(filter_data_len) if filter_data_len else ''}"
    f"{'_filter-target-' + str(filter_target_len) if filter_target_len else ''}"
    f"_accum-{accum_grad_multiple_step}{'_rand-seed-' + str(random_seed) if random_seed is not None else ''}"
    f"{'_no-self-att-until-' + str(disable_enc_self_att_until_epoch) if disable_enc_self_att_until_epoch is not None else ''}"
    f"{'_cut-init-sil' if cutoff_initial_silence else ''}"
  )

  if lr_scheduling_opts is None:
    lr_scheduling_opts = {"type": "dyn_lr_piecewise_linear_epoch-wise_v2", "peak_lr": 1e-3}
    lr_scheduling_opts["init_lr"] = lr_scheduling_opts["peak_lr"] * 1e-2

  alias += (
    f"/lr_{'_'.join([f'{k}-{v}' if k != 'type' else f'{v}' for k, v in lr_scheduling_opts.items()])}"
  )

  if ctc_aux_loss_layers:
    alias += f"_ctc-aux-{'-'.join(map(str, ctc_aux_loss_layers))}"
  if ctc_aux_loss_scales:
    alias += f"_scales-{'-'.join(map(str, ctc_aux_loss_scales))}"
  if ctc_aux_loss_focal_loss_factors:
    alias += f"_aux-focal-loss-{'-'.join(map(str, ctc_aux_loss_focal_loss_factors))}"

  reg_opts = copy.deepcopy(regularization_opts[regularization_type])

  train_opts = {
    "dataset_opts": {
      "use_speed_pert": use_speed_pert,
      "use_speed_pert_w_flip": use_speed_pert_w_flip,
      "cutoff_initial_silence": cutoff_initial_silence,
    },
    # "import_model_train_epoch1": None,
    "accum_grad_multiple_step": accum_grad_multiple_step,
    "pos_emb_dropout": 0.1,
    "rf_att_dropout_broadcast": False,
    "batch_size": batch_size,
    "batching": "laplace:.1000",
    "aux_loss_layers": ctc_aux_loss_layers,
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
      "weight_decay": reg_opts.pop("weight_decay"),
    },
    "cleanup_old_models": {"keep": keep_epochs if keep_epochs else [n_epochs], "keep_best_n": 4, "keep_last_n": 1},
    **reg_opts,
  }

  if filter_data_len:
    train_opts["max_seq_length"] = {"data": filter_data_len}

  if filter_target_len:
    train_opts["max_seq_length"] = {"targets": filter_target_len}

  if gpu_mem_rqmt == 24 and use_torch_amp:
    train_opts["torch_amp"] = "bfloat16"

  if random_seed is not None:
    train_opts["random_seed"] = random_seed

  if disable_enc_self_att_until_epoch is not None:
    train_opts["disable_enc_self_att_until_epoch"] = disable_enc_self_att_until_epoch

  if checkpoint_alias is not None:
    train_opts["preload_from_files"] = {
      "pretrained_params": {
        "filename": external_checkpoints[checkpoint_alias] if checkpoint_path is None else checkpoint_path,
        "init_for_train": True,
        "ignore_missing": True,  # because of length model params
      }
    }

  lr_scheduling_type = lr_scheduling_opts.pop("type")
  if lr_scheduling_type == "dyn_lr_piecewise_linear":
    train_opts["lr_opts"] = {
      "type": "dyn_lr_piecewise_linear",
      "batch_size": batch_size,
      "num_epochs": n_epochs,
      **lr_scheduling_opts,
    }
  elif lr_scheduling_type == "const":
    train_opts["lr_opts"] = {
      "type": "const",
      "const_lr": 1e-6,
    }
  elif lr_scheduling_type.startswith("dyn_lr_piecewise_linear_epoch-wise"):
    train_opts["lr_opts"] = {
      "type": lr_scheduling_type,
      "num_epochs": n_epochs,
      **lr_scheduling_opts,
    }
  else:
    assert lr_scheduling_type == "const_then_linear"
    train_opts["lr_opts"] = {
      "type": "const_then_linear",
      "const_lr": 1e-4,
      "const_frac": 1 / 3,
      "final_lr": 1e-6,
      "num_epochs": n_epochs
    }

  if use_curriculum_learning:
    train_opts["dataset_opts"]["epoch_wise_filter"] = {(1, 5): {"max_mean_len": 1000}}

  if ctc_aux_loss_layers:
    train_opts["aux_loss_type"] = "ce"
  if ctc_aux_loss_focal_loss_factors:
    train_opts["aux_loss_focal_loss_factors"] = ctc_aux_loss_focal_loss_factors
  if ctc_aux_loss_scales:
    train_opts["aux_loss_scales"] = ctc_aux_loss_scales

  train_rqmt = {
    "time": time_rqmt,
    "gpu_mem": gpu_mem_rqmt,
  }
  if use_mgpu:
    train_rqmt.update({
      "horovod_num_processes": 4,
      "distributed_launch_cmd": "torchrun"
    })
    train_opts["torch_distributed"] = {"reduce_type": "param", "param_sync_step": 100}

  train_rqmt["sbatch_args"] = []
  if cluster_reservation_string:
    train_rqmt["sbatch_args"] += ["--reservation", cluster_reservation_string]
  train_rqmt["sbatch_args"] += ["--exclude", "cn-222"]

  return train_opts, train_rqmt, alias
