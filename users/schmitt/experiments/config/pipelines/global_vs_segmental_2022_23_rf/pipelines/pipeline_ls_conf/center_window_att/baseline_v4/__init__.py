from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from . import (
  baseline,
  att_weight_feedback,
  full_label_context,
  label_context1,
)


def run_exps():
  # Done
  baseline.run_exps()
  # Done
  att_weight_feedback.run_exps()
  # Done
  full_label_context.run_exps()
  # Done
  label_context1.run_exps()

  # --------------------------- full-sum training exps ---------------------------

  # for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #         win_size_list=(5,),
  #         label_decoder_state="nb-lstm",
  #         use_att_ctx_in_state=False,
  #         use_weight_feedback=False,
  # ):
  #   for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(125,),
  #     use_speed_pert=True,
  #     batch_size=8_000,
  #     time_rqmt=80,
  #     use_mgpu=False,
  #     beam_size=4,
  #     lattice_downsampling=1,
  #     alignment_interpolation_factor=0.5,
  #   ):
  #     for epoch, chckpt in checkpoint["checkpoints"].items():
  #       realign.center_window_returnn_realignment(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=chckpt,
  #         checkpoint_alias=f"epoch-{epoch}",
  #         plot=True,
  #       )
  #
  #   for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(125,),
  #     use_speed_pert=True,
  #     batch_size=8_000,
  #     time_rqmt=80,
  #     use_mgpu=False,
  #     beam_size=100,
  #     lattice_downsampling=3,
  #     alignment_interpolation_factor=0.0,
  #   ):
  #     for epoch, chckpt in checkpoint["checkpoints"].items():
  #       realign.center_window_returnn_realignment(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=chckpt,
  #         checkpoint_alias=f"epoch-{epoch}",
  #         plot=True,
  #       )
  #
  #   for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(125,),
  #     use_speed_pert=True,
  #     batch_size=8_000,
  #     time_rqmt=80,
  #     use_mgpu=False,
  #     beam_size=1,
  #     lattice_downsampling=1,
  #     alignment_interpolation_factor=0.0,
  #     train_on_viterbi_paths=True,
  #   ):
  #     for epoch, chckpt in checkpoint["checkpoints"].items():
  #       realign.center_window_returnn_realignment(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=chckpt,
  #         checkpoint_alias=f"epoch-{epoch}",
  #         plot=True,
  #       )
  #
  # for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #         win_size_list=(15,),
  #         label_decoder_state="nb-lstm",
  #         use_att_ctx_in_state=False,
  #         use_weight_feedback=False,
  # ):
  #   for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(125,),
  #     use_speed_pert=True,
  #     batch_size=8_000,
  #     time_rqmt=80,
  #     use_mgpu=False,
  #     beam_size=100,
  #     lattice_downsampling=8,
  #     alignment_interpolation_factor=0.0,
  #   ):
  #     for epoch, chckpt in checkpoint["checkpoints"].items():
  #       if epoch > 58:
  #         continue
  #       realign.center_window_returnn_realignment(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=chckpt,
  #         checkpoint_alias=f"epoch-{epoch}",
  #         plot=True,
  #       )
  #
  # for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #         win_size_list=(5,),
  #         label_decoder_state="nb-lstm",
  #         use_att_ctx_in_state=False,
  #         use_weight_feedback=False,
  #         bpe_vocab_size=1056,
  # ):
  #   for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(125,),
  #     use_speed_pert=True,
  #     batch_size=8_000,
  #     time_rqmt=80,
  #     use_mgpu=False,
  #   ):
  #     for epoch, chckpt in checkpoint["checkpoints"].items():
  #       realign.center_window_returnn_realignment(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=chckpt,
  #         checkpoint_alias=f"epoch-{epoch}",
  #       )
