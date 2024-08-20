from typing import Tuple, Optional, List
import itertools

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechGlobalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.train import get_common_train_opts_rqmt
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.train import _returnn_v2_train_step, from_scratch_training
from i6_experiments.users.schmitt.custom_load_params import load_missing_params

from i6_core.returnn.training import PtCheckpoint


def train_global_att(
        alias: str,
        config_builder: LibrispeechGlobalAttConformerConfigBuilderRF,
        n_epochs: int,
        batch_size: int = 15_000,
        time_rqmt: int = 168,
        use_mgpu: bool = True,
        ce_aux_loss_layers: Optional[Tuple[int, ...]] = None,
        ce_aux_loss_focal_loss_factors: Optional[Tuple[float, ...]] = None,
        ce_aux_loss_scales: Optional[Tuple[float, ...]] = None,
        use_curriculum_learning: bool = True,
        lr_scheduling_type: str = "dyn_lr_piecewise_linear",
        regularization_type: str = "v1",
        checkpoint_alias: Optional[str] = None,
        checkpoint_path: Optional[PtCheckpoint] = None,
        use_speed_pert: bool = True,
):
  # alias += (
  #         f"/train_from_scratch/{n_epochs}-epochs_wo-ctc-loss"
  #         f"{'_mgpu-4' if use_mgpu else ''}_bs-{batch_size}"
  # )

  train_opts, train_rqmt, alias_ = get_common_train_opts_rqmt(
    n_epochs=n_epochs,
    time_rqmt=time_rqmt,
    batch_size=batch_size,
    use_mgpu=use_mgpu,
    ce_aux_loss_layers=ce_aux_loss_layers,
    ce_aux_loss_focal_loss_factors=ce_aux_loss_focal_loss_factors,
    ce_aux_loss_scales=ce_aux_loss_scales,
    use_curriculum_learning=use_curriculum_learning,
    lr_scheduling_type=lr_scheduling_type,
    regularization_type=regularization_type,
    use_speed_pert=use_speed_pert,
    training_type="train",
    checkpoint_alias=checkpoint_alias,
    checkpoint_path=checkpoint_path,
  )

  alias += alias_

  train_opts.update({
    "train_step_func": _returnn_v2_train_step,
    "train_def": from_scratch_training,
  })

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
