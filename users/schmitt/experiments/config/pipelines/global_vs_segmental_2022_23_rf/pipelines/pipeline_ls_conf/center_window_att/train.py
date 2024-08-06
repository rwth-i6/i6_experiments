import copy
from typing import Tuple, Optional, List, Dict, Union, Callable

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechSegmentalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.train import (
  _returnn_v2_train_step,
  _returnn_v2_full_sum_train_step,
  viterbi_training,
  full_sum_training,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import (
  external_checkpoints,
  default_import_model_name,
)
from i6_experiments.users.schmitt.custom_load_params import load_missing_params
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE10025_CTC_ALIGNMENT

from i6_core.returnn.training import PtCheckpoint


def _get_optimizer_alias(optimizer_opts: Dict):
  return (
    f"opt-{optimizer_opts['class']}-eps-{optimizer_opts['epsilon']}-wd-{optimizer_opts.get('weight_decay', 0.0)}"
  )


def _get_reduced_input_len(input_len: int, config_builder: LibrispeechSegmentalAttConformerConfigBuilderRF):
  return int(input_len - config_builder.red_subtrahend + config_builder.red_factor - 1) // config_builder.red_factor


def train_center_window_att_viterbi_from_scratch(
        alias: str,
        config_builder: LibrispeechSegmentalAttConformerConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 80,
        batch_size: int = 15_000,
        use_mgpu: bool = True,
        chunked_data_len: Optional[int] = None,
        nb_loss_scale: float = 1.0,
        b_loss_scale: float = 1.0,
        do_realignments: bool = False,
        ce_aux_loss_layers: Optional[Tuple[int, ...]] = None,
        ce_aux_loss_focal_loss_factors: Optional[Tuple[float, ...]] = None,
        ce_aux_loss_scales: Optional[Tuple[float, ...]] = None,
        use_curriculum_learning: bool = True,
        lr_scheduling_type: str = "dyn_lr_piecewise_linear",
):
  for n_epochs in n_epochs_list:
    alias += (
      f"/{'viterbi' if do_realignments else 'fixed-path'}-train_from_scratch/{n_epochs}-ep_bs-{batch_size}"
      f"{'_mgpu-4' if use_mgpu else ''}_wo-speed-pert"
      f"_{'chunked-data-len-' + str(chunked_data_len) if chunked_data_len else 'no-chunking'}/"
      f"_nb-loss-x{nb_loss_scale}_b-loss-x{b_loss_scale}"
      f"_ce-aux-{'-'.join(map(str, ce_aux_loss_layers)) if ce_aux_loss_layers else 'None'}"
      f"_scales-{'-'.join(map(str, ce_aux_loss_scales)) if ce_aux_loss_scales else 'None'}"
      f"_aux-focal-loss-{'-'.join(map(str, ce_aux_loss_focal_loss_factors)) if ce_aux_loss_focal_loss_factors else 'None'}"
      f"_{'curriculum' if use_curriculum_learning else 'no-curriculum'}"
      f"_lr-{lr_scheduling_type}"
    )

    train_opts = {
      "dataset_opts": {
        "use_speed_pert": False,  # not implemented yet for fixed-path training
        # "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}},
        "hdf_targets": LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths,
      },
      # "import_model_train_epoch1": None,
      "accum_grad_multiple_step": 4,
      "pos_emb_dropout": 0.1,
      "rf_att_dropout_broadcast": False,
      "batch_size": batch_size,
      "batching": "laplace:.1000",
      "aux_loss_layers": ce_aux_loss_layers,
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
      "nb_loss_scale": nb_loss_scale,
      "b_loss_scale": b_loss_scale,
      "training_do_realignments": do_realignments,
    }

    if lr_scheduling_type == "dyn_lr_piecewise_linear":
      train_opts["lr_opts"] = {
        "type": "dyn_lr_piecewise_linear",
        "batch_size": batch_size,
        "num_epochs": n_epochs,
        "learning_rate": 1e-3,
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

    if ce_aux_loss_layers:
      train_opts["aux_loss_type"] = "ce"
    if ce_aux_loss_focal_loss_factors:
      train_opts["aux_loss_focal_loss_factors"] = ce_aux_loss_focal_loss_factors
    if ce_aux_loss_scales:
      train_opts["aux_loss_scales"] = ce_aux_loss_scales

    if chunked_data_len:
      train_opts.update({
        "chunking": (
          {
            "data": chunked_data_len + config_builder.red_subtrahend,
            "targets": _get_reduced_input_len(chunked_data_len, config_builder)
          },
          {"data": chunked_data_len // 2, "targets": _get_reduced_input_len(chunked_data_len // 2, config_builder)},
        ),
        "min_chunk_size": {"data": config_builder.red_subtrahend + 1, "targets": 1}
      })

    train_rqmt = {
      "time": time_rqmt,
    }
    if use_mgpu:
      train_rqmt.update({
        "horovod_num_processes": 4,
        "distributed_launch_cmd": "torchrun"
      })
      train_opts["torch_distributed"] = {"reduce_type": "param", "param_sync_step": 100}

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt=train_rqmt,
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
    yield alias, checkpoint


def train_center_window_att_full_sum_from_scratch(
        alias: str,
        config_builder: LibrispeechSegmentalAttConformerConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        time_rqmt: int = 80,
        gpu_mem_rqmt: int = 11,
        use_speed_pert: bool = False,
        batch_size: int = 15_000,
        use_mgpu: bool = True,
        beam_size: Optional[int] = None,
        lattice_downsampling: int = 1,
        alignment_interpolation_factor: float = 0.5,
        train_on_viterbi_paths: bool = False,
        only_use_blank_model: bool = False,
        checkpoint_alias: Optional[str] = None,
        lr_scheduling_type: str = "dyn_lr_piecewise_linear",
        checkpoint_path: Optional[PtCheckpoint] = None,
):
  for n_epochs in n_epochs_list:
    alias += (
      f"/full-sum-train_from_{'scratch' if checkpoint_alias is None else checkpoint_alias}/{n_epochs}-epochs_bs-{batch_size}"
      f"{'_mgpu-4' if use_mgpu else ''}_{'w' if use_speed_pert else 'wo'}-sp"
      f"_beams-{beam_size}_lat-down-{lattice_downsampling}_{alignment_interpolation_factor}-interp"
      f"_{'ce' if train_on_viterbi_paths else 'sum'}-loss_lr-{lr_scheduling_type}"
    )

    train_opts = {
      "dataset_opts": {
        "use_speed_pert": use_speed_pert,
        "hdf_targets": {},  # do not use alignment for full sum training
        "seq_postfix": None,
        "target_is_alignment": False,
      },
      # "import_model_train_epoch1": None,
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
      # "max_seq_length": {"targets": 75},
      "train_def": full_sum_training,
      "train_step_func": _returnn_v2_full_sum_train_step,
    }

    if checkpoint_alias is not None:
      train_opts["preload_from_files"] = {
        "pretrained_global_att_params": {
          "filename": external_checkpoints[checkpoint_alias] if checkpoint_path is None else checkpoint_path,
          "init_for_train": True,
          "ignore_missing": True,  # because of length model params
        }
      }
    else:
      train_opts["dataset_opts"]["epoch_wise_filter"] = {(1, 5): {"max_mean_len": 1000}}

    if lr_scheduling_type == "dyn_lr_piecewise_linear":
      train_opts["lr_opts"] = {
        "type": "dyn_lr_piecewise_linear",
        "batch_size": batch_size,
        "num_epochs": n_epochs,
        "learning_rate": 1e-3,
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

    full_sum_training_opts = {
      "alignment_interpolation_factor": alignment_interpolation_factor,
      "lattice_downsampling": lattice_downsampling,
      "only_use_blank_model": only_use_blank_model,
      "beam_size": beam_size,
      "train_on_viterbi_paths": train_on_viterbi_paths,
    }

    train_opts["full_sum_training_opts"] = full_sum_training_opts

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

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      num_epochs=n_epochs,
      train_rqmt=train_rqmt,
      train_opts=train_opts
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    checkpoint = {
      "model_dir": model_dir,
      "learning_rates": learning_rates,
      "key": "dev_loss_full_sum_loss",
      "checkpoints": checkpoints,
      "n_epochs": n_epochs
    }
    yield alias, checkpoint


def train_center_window_att_viterbi_import_global_tf(
        alias: str,
        config_builder: LibrispeechSegmentalAttConformerConfigBuilderRF,
        n_epochs_list: Tuple[int, ...],
        const_lr_list: Tuple[float, ...] = (1e-4,),
        time_rqmt: int = 80,
        alignment_augmentation_opts: Optional[Dict] = None,
        import_model_name: str = default_import_model_name,
        keep_best_n: int = 4,
        nb_loss_scale_list: Tuple[float, ...] = (1.0,),
        b_loss_scale_list: Tuple[float, ...] = (1.0,),
        optimizer_opts: Optional[Dict] = None,
        reset_eos_params: bool = False,
        batch_size: int = 15_000,
        accum_grad_multiple_step: int = 2,
        specaugment_steps: Tuple[int, ...] = (0, 1000, 2000),
        gradient_clip_global_norm: float = 0.0,
        rf_att_dropout_broadcast: bool = False,
        pos_emb_dropout: float = 0.1,
):
  if not config_builder.use_att_ctx_in_state and "lstm" in config_builder.label_decoder_state:
    # only randomly init FF weights, since only the input dim of the lstm layer is different
    custom_missing_load_func = load_missing_params
  else:
    custom_missing_load_func = None

  if optimizer_opts is None:
    optimizer_opts = {"class": "adam", "epsilon": 1e-8}
    # optimizer_opts = {
    #   "class": "adamw",
    #   "epsilon": 1e-16,
    #   "weight_decay": 0.01,
    #   "weight_decay_modules_blacklist": [
    #       "rf.Embedding",
    #       "rf.LearnedRelativePositionalEncoding",
    #   ],
    # }

  for n_epochs in n_epochs_list:
    for const_lr in const_lr_list:
      for nb_loss_scale in nb_loss_scale_list:
        for b_loss_scale in b_loss_scale_list:
          train_alias = alias + (
            f"/train_from_{import_model_name}/standard-training/{n_epochs}-ep_{const_lr}-const-lr_bs-{batch_size}"
            f"_nb-loss-x{nb_loss_scale}_b-loss-x{b_loss_scale}_{_get_optimizer_alias(optimizer_opts)}"
            f"{'_reset-eos' if reset_eos_params else ''}"
          )
          if alignment_augmentation_opts:
            opts = alignment_augmentation_opts
            train_alias += f"_align-aug-{opts['num_iterations']}-iters_{opts['max_shift']}-max-shift"

          train_opts = {
            "dataset_opts": {
              "hdf_targets": LibrispeechBPE10025_CTC_ALIGNMENT.alignment_paths,
            },
            "preload_from_files": {
              "pretrained_global_att_params": {
                "filename": external_checkpoints[import_model_name],
                "init_for_train": True,
                "ignore_missing": True,  # because of length model params
              }
            },
            "aux_loss_layers": None,
            "accum_grad_multiple_step": 2,
            "optimizer": optimizer_opts,
            "train_def": viterbi_training,
            "train_step_func": _returnn_v2_train_step,
            "batching": "random",
            "batch_size": batch_size,
            "lr_opts": {
              "type": "const_then_linear",
              "const_lr": const_lr,
              "const_frac": 1 / 3,
              "final_lr": 1e-6,
              "num_epochs": n_epochs
            },
            "alignment_augmentation_opts": alignment_augmentation_opts,
            "nb_loss_scale": nb_loss_scale,
            "b_loss_scale": b_loss_scale,
            "reset_eos_params": reset_eos_params,
          }
          if custom_missing_load_func:
            train_opts["preload_from_files"]["pretrained_global_att_params"]["custom_missing_load_func"] = custom_missing_load_func
          train_opts["cleanup_old_models"] = {"keep_best_n": keep_best_n, "keep_last_n": 1, "keep": [n_epochs]}

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
