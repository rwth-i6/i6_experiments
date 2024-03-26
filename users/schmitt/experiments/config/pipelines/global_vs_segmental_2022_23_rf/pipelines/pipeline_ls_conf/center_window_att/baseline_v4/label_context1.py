from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import (
  default_import_model_name,
)


def run_exps():
  # -------------------------- fixed-path training from global att checkpoint (Done) --------------------------------

  for win_size in (1, 5):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(win_size,),
            label_decoder_state="nb-2linear-ctx1",
            use_weight_feedback=False,
            use_att_ctx_in_state=False
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(300,),
              const_lr_list=(1e-4,),
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          lm_type="trafo",
          lm_scale_list=(0.54,),
          ilm_type="mini_att",
          ilm_scale_list=(0.4,),
          subtract_ilm_eos_score=True,
          use_recombination="sum",
          corpus_keys=("dev-other", "test-other"),
          beam_size_list=(12, 84),
        )
        if win_size == 5:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            run_analysis=True,
            att_weight_seq_tags=[
              "dev-other/116-288045-0017/116-288045-0017",
              "dev-other/116-288045-0014/116-288045-0014",
            ]
          )

  # -------------------------- from-scratch fixed-path training (similar to Atanas) (Done) ----------------------------

  # for label_decoder_state in ("nb-2linear-ctx1",):
  #   for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
  #           win_size_list=(1,),
  #           label_decoder_state=label_decoder_state,
  #           use_weight_feedback=False,
  #           use_att_ctx_in_state=False
  #   ):
  #     for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
  #       alias=model_alias,
  #       config_builder=config_builder,
  #       n_epochs_list=(600,),
  #       chunked_data_len=95_440,  # -> 600ms -> 100 encoder frames
  #       nb_loss_scale=6.0,
  #       use_mgpu=False,
  #     ):
  #       recog.center_window_returnn_frame_wise_beam_search(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=checkpoint,
  #       )

  for label_decoder_state in ("nb-2linear-ctx1",):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(1,),
            label_decoder_state=label_decoder_state,
            use_weight_feedback=False,
            use_att_ctx_in_state=False
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(600,),
        nb_loss_scale=6.0,
        use_mgpu=False,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          lm_type="trafo",
          lm_scale_list=(0.6,),
          ilm_type="mini_att",
          ilm_scale_list=(0.3,),
          subtract_ilm_eos_score=True,
          use_recombination="sum",
          corpus_keys=("dev-other", "test-other"),
          beam_size_list=(12, 84),
        )

  for label_decoder_state in ("nb-2linear-ctx1",):
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(1,),
            label_decoder_state=label_decoder_state,
            use_weight_feedback=False,
            use_att_ctx_in_state=False
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(600,),
        nb_loss_scale=6.0,
        use_mgpu=False,
        ce_aux_loss_layers=(6, 12),
        ce_aux_loss_focal_loss_factors=(1.0, 1.0),
        ce_aux_loss_scales=(0.3, 1.0)
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          lm_type="trafo",
          lm_scale_list=(0.6,),
          ilm_type="mini_att",
          ilm_scale_list=(0.3,),
          subtract_ilm_eos_score=True,
          use_recombination="sum",
          corpus_keys=("dev-other", "test-other"),
          beam_size_list=(12, 84),
        )

  # -------------------------- from-scratch Viterbi training --------------------------------

  # for label_decoder_state in ("nb-2linear-ctx1",):
  #   for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
  #           win_size_list=(1,),
  #           label_decoder_state=label_decoder_state,
  #           use_weight_feedback=False,
  #           use_att_ctx_in_state=False
  #   ):
  #     for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
  #       alias=model_alias,
  #       config_builder=config_builder,
  #       n_epochs_list=(600,),
  #       nb_loss_scale=6.0,
  #       use_mgpu=False,
  #       do_realignments=True,
  #     ):
  #       recog.center_window_returnn_frame_wise_beam_search(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=checkpoint,
  #       )

  # -------------------------- full-sum training --------------------------------

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          label_decoder_state="nb-2linear-ctx1",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=5048,
  ):
    # -------------------------- from-scratch (5k) --------------------------------
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      use_speed_pert=True,
      batch_size=3_000,
      time_rqmt=168,
      use_mgpu=True,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          realign.center_window_returnn_realignment(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_alias=f"epoch-{epoch}",
            plot=True,
          )
        if epoch == 0:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
          )

    # -------------------------- from-global-att (bpe 5k) --------------------------------
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            use_speed_pert=True,
            batch_size=3_000,
            time_rqmt=80,
            checkpoint_alias="luca-aed-bpe5k",
            lr_scheduling_type="const_then_linear",
            use_mgpu=False
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          realign.center_window_returnn_realignment(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_alias=f"epoch-{epoch}",
            plot=True,
          )
        if epoch in (30, 128):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
          )

  # -------------------------- from-global-att (bpe 10k) --------------------------------
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          label_decoder_state="nb-2linear-ctx1",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(300,),
      use_speed_pert=True,
      batch_size=6_000,
      time_rqmt=48,
      checkpoint_alias=default_import_model_name,
      lr_scheduling_type="const_then_linear",
      gpu_mem_rqmt=24,
      use_mgpu=False
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          realign.center_window_returnn_realignment(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_alias=f"epoch-{epoch}",
            plot=True,
          )

  # -------------------------- from-scratch (1k) --------------------------------
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          label_decoder_state="nb-2linear-ctx1",
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          bpe_vocab_size=1056,
  ):
    # -------------------------- from-scratch --------------------------------
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      use_speed_pert=True,
      batch_size=8_000,
      time_rqmt=168,
      use_mgpu=True,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          realign.center_window_returnn_realignment(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_alias=f"epoch-{epoch}",
            plot=True,
          )

        if epoch == 43:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
          )