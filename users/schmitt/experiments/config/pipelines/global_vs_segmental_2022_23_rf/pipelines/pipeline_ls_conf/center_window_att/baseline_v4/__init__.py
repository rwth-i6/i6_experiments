from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), label_decoder_state="nb-linear1", use_att_ctx_in_state=False, use_weight_feedback=False,
  ):
    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        import_model_name=import_model_name,
      ):
        for recombination in (None,):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            # checkpoint_aliases=("last",),
            use_recombination=recombination,
          )

  # for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #   win_size_list=(5,), label_decoder_state="joint-lstm",
  # ):
  #   for import_model_name in ("glob.conformer.mohammad.5.4",):
  #     for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
  #       alias=model_alias,
  #       config_builder=config_builder,
  #       n_epochs_list=(100,),
  #       import_model_name=import_model_name,
  #     ):
  #       for recombination in (None,):
  #         recog.center_window_returnn_frame_wise_beam_search(
  #           alias=train_alias,
  #           config_builder=config_builder,
  #           checkpoint=checkpoint,
  #           # checkpoint_aliases=("last",),
  #           use_recombination=recombination,
  #         )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="nb-lstm",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=8_000,
      time_rqmt=80,
      use_mgpu=False,
      beam_size=4,
      lattice_downsampling=1,
      alignment_interpolation_factor=0.5,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        realign.center_window_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=chckpt,
          checkpoint_alias=f"epoch-{epoch}",
          plot=True,
        )

    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=8_000,
      time_rqmt=80,
      use_mgpu=False,
      beam_size=100,
      lattice_downsampling=3,
      alignment_interpolation_factor=0.0,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        realign.center_window_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=chckpt,
          checkpoint_alias=f"epoch-{epoch}",
          plot=True,
        )

    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=8_000,
      time_rqmt=80,
      use_mgpu=False,
      beam_size=1,
      lattice_downsampling=1,
      alignment_interpolation_factor=0.0,
      train_on_viterbi_paths=True,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        realign.center_window_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=chckpt,
          checkpoint_alias=f"epoch-{epoch}",
          plot=True,
        )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(15,),
          label_decoder_state="nb-lstm",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=8_000,
      time_rqmt=80,
      use_mgpu=False,
      beam_size=100,
      lattice_downsampling=8,
      alignment_interpolation_factor=0.0,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        realign.center_window_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=chckpt,
          checkpoint_alias=f"epoch-{epoch}",
          plot=True,
        )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="nb-lstm",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=1056,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=8_000,
      time_rqmt=80,
      use_mgpu=False,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        realign.center_window_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=chckpt,
          checkpoint_alias=f"epoch-{epoch}",
        )
