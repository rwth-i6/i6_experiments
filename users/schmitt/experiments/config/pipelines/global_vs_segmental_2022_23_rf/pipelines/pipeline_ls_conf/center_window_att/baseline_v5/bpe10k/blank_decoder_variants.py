from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------- blank decoder v4 (label ctx 1) (Done) ---------------------

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1, 5,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
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
        run_analysis=True,
        analyze_gradients=True,
      )
      for concat_num in (8,):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          run_analysis=True,
          analyze_gradients=True,
          concat_num=concat_num,
          batch_size=30_000,
          att_weight_seq_tags=[
            "dev-other/116-288045-0008/116-288045-0008;dev-other/116-288045-0009/116-288045-0009;dev-other/116-288045-0010/116-288045-0010;dev-other/116-288045-0011/116-288045-0011;dev-other/116-288045-0012/116-288045-0012;dev-other/116-288045-0013/116-288045-0013;dev-other/116-288045-0014/116-288045-0014;dev-other/116-288045-0015/116-288045-0015",
          ]
        )

  # same as above, just doing recog during training for some epochs (Done)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in (10, 50, 100, 150, 200, 250):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
          )

      for corpus_key in [
        "dev-other_0.1-5.1",
        "dev-other_5.1-10.1",
        "dev-other_10.1-15.1",
        "dev-other_15.1-20.1",
        "dev-other_20.1-25.1",
        "dev-other_25.1-30.1",
        "dev-other_30.1-35.1",
      ]:
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          corpus_keys=(corpus_key,),
        )

  # with large window (Done)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(499,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
            batch_size=10_000,
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=8,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
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

  # ----------------- first fixed-path, then full-sum (Running) -----------------
  fixed_path_checkpoint_ep300 = None
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
  ):
    for fixed_path_train_alias, fixed_path_checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      fixed_path_checkpoint_ep300 = fixed_path_checkpoint["checkpoints"][300]
      recog.center_window_returnn_frame_wise_beam_search(
        alias=fixed_path_train_alias,
        config_builder=config_builder,
        checkpoint=fixed_path_checkpoint,
      )

  # only current frame in readout (Running)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          use_correct_dim_tags=True,
  ):
    for full_sum_train_alias, full_sum_checkpoint in train.train_center_window_att_full_sum_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            use_speed_pert=True,
            batch_size=5_000,
            time_rqmt=80,
            gpu_mem_rqmt=24,
            checkpoint_alias="ctc-fixed-path-300ep",
            lr_scheduling_type="const_then_linear",
            use_mgpu=False,
            checkpoint_path=fixed_path_checkpoint_ep300
    ):
      for epoch, chckpt in full_sum_checkpoint["checkpoints"].items():
        if epoch % 50 == 0:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=full_sum_train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
          )
        if epoch == 109:
          realign.center_window_returnn_realignment(
            alias=full_sum_train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_alias=f"epoch-{epoch}",
            plot=True,
          )

  # ---------------------------------------------------------------------------------------------------

  # global att + current frame in readout (Running)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(None,),
          blank_decoder_version=4,
          use_att_ctx_in_state=False,
          use_weight_feedback=False,
          use_correct_dim_tags=True,
          use_current_frame_in_readout=True,
  ):
    for full_sum_train_alias, full_sum_checkpoint in train.train_center_window_att_full_sum_from_scratch(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            use_speed_pert=True,
            batch_size=5_000,
            time_rqmt=80,
            gpu_mem_rqmt=24,
            checkpoint_alias="ctc-fixed-path-300ep",
            lr_scheduling_type="const_then_linear",
            use_mgpu=False,
            checkpoint_path=fixed_path_checkpoint_ep300
    ):
      for epoch, chckpt in full_sum_checkpoint["checkpoints"].items():
        if epoch % 50 == 0:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=full_sum_train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
          )

  # ------------------- blank decoder v4 (label ctx 1) fixed-path from-scratch (Done) ---------------------

  # window size 1 (like transducer)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,),
          use_weight_feedback=False,
          use_att_ctx_in_state=False,
          blank_decoder_version=4,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      nb_loss_scale=6.0,
      use_mgpu=True,
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
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
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
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        analyze_gradients=True,
        run_analysis=True,
      )

  # window size 11 + current frame in readout (Done)
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(11,),
          use_weight_feedback=False,
          use_att_ctx_in_state=False,
          blank_decoder_version=4,
          use_current_frame_in_readout=True,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      nb_loss_scale=6.0,
      use_mgpu=True,
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
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
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
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
      )
