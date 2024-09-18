from typing import Tuple, Optional, List, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechCtcAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import GlobalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.train import get_common_train_opts_rqmt
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.train import _returnn_v2_train_step, from_scratch_training

from i6_core.returnn.training import PtCheckpoint


def train_ctc(
        alias: str,
        config_builder: LibrispeechCtcAttConformerConfigBuilderRF,
        n_epochs: int,
        batch_size: int = 15_000,
        time_rqmt: int = 168,
        gpu_mem_rqmt: int = 11,
        use_mgpu: bool = True,
        ctc_aux_loss_layers: Optional[Tuple[int, ...]] = None,
        ctc_aux_loss_focal_loss_factors: Optional[Tuple[float, ...]] = None,
        ctc_aux_loss_scales: Optional[Tuple[float, ...]] = None,
        use_curriculum_learning: bool = True,
        lr_scheduling_opts: Optional[Dict] = None,
        regularization_type: str = "v1",
        checkpoint_alias: Optional[str] = None,
        checkpoint_path: Optional[PtCheckpoint] = None,
        use_speed_pert: bool = True,
        keep_epochs: Optional[List[int]] = None,
        filter_data_len: Optional[float] = None,
        filter_target_len: Optional[float] = None,
        cluster_reservation_string: Optional[str] = None,
        accum_grad_multiple_step: int = 4,
        use_torch_amp: bool = False,
        random_seed: Optional[int] = None,
        disable_enc_self_att_until_epoch: Optional[int] = None,
):

  train_opts, train_rqmt, alias_ = get_common_train_opts_rqmt(
    n_epochs=n_epochs,
    time_rqmt=time_rqmt,
    batch_size=batch_size,
    use_mgpu=use_mgpu,
    ctc_aux_loss_layers=ctc_aux_loss_layers,
    ctc_aux_loss_focal_loss_factors=ctc_aux_loss_focal_loss_factors,
    ctc_aux_loss_scales=ctc_aux_loss_scales,
    use_curriculum_learning=use_curriculum_learning,
    lr_scheduling_opts=lr_scheduling_opts,
    regularization_type=regularization_type,
    use_speed_pert=use_speed_pert,
    training_type="train",
    checkpoint_alias=checkpoint_alias,
    checkpoint_path=checkpoint_path,
    gpu_mem_rqmt=gpu_mem_rqmt,
    keep_epochs=keep_epochs,
    filter_data_len=filter_data_len,
    filter_target_len=filter_target_len,
    cluster_reservation_string=cluster_reservation_string,
    accum_grad_multiple_step=accum_grad_multiple_step,
    use_torch_amp=use_torch_amp,
    random_seed=random_seed,
    disable_enc_self_att_until_epoch=disable_enc_self_att_until_epoch,
  )

  alias += alias_

  train_opts.update({
    "train_step_func": _returnn_v2_train_step,
    "train_def": from_scratch_training,
  })

  train_opts["dataset_opts"].update({
    "target_is_alignment": False,
    "seq_postfix": None,
    "use_multi_proc": True,
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
