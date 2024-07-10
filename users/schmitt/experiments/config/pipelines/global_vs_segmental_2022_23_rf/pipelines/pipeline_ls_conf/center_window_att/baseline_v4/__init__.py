from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)


def run_exps():
  # --------------------------- Should be like chunked AED with chunk-size 1 ---------------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(1,),
          label_decoder_state="joint-lstm",
          use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(100,),
            const_lr_list=(1e-4,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        use_recombination="sum",
      )

  # --------------------------- For comparison with the above with larger win-size ---------------------------

  for use_att_dep in (True, False):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
            win_size_list=(5,),
            label_decoder_state="joint-lstm",
            use_weight_feedback=use_att_dep,
            use_att_ctx_in_state=use_att_dep
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
          use_recombination=None,
        )
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

  # --------------------------- Comparison with Atanas' transducer models ---------------------------

  for label_decoder_state in ("nb-lstm", "nb-2linear-ctx1"):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
            win_size_list=(5,),
            label_decoder_state=label_decoder_state,
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

  for label_decoder_state in ("nb-lstm", "nb-2linear-ctx1"):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
            win_size_list=(1,),
            label_decoder_state=label_decoder_state,
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

  # --------------------------- Comparison with Atanas' transducer models (reset EOS params) ------------------------

  for label_decoder_state in ("nb-lstm",):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
            win_size_list=(1,),
            label_decoder_state=label_decoder_state,
            use_weight_feedback=False,
            use_att_ctx_in_state=False
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(300,),
        const_lr_list=(1e-4,),
        reset_eos_params=True,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          reset_eos_params=True,
        )

  # -------------------------- test separating blank from the softmax --------------------------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="nb-2linear-ctx1",
          use_weight_feedback=False,
          use_att_ctx_in_state=False,
          separate_blank_from_softmax=True
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(300,),
      const_lr_list=(1e-4,),
      keep_best_n=0,
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
      )

  # --------------------------- test different blank and non-blank loss scales ---------------------------

  # for label_decoder_state in ("nb-2linear-ctx1",):
  #   for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #           win_size_list=(1,),
  #           label_decoder_state=label_decoder_state,
  #           use_weight_feedback=False,
  #           use_att_ctx_in_state=False
  #   ):
  #     for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
  #       alias=model_alias,
  #       config_builder=config_builder,
  #       n_epochs_list=(10,),
  #       const_lr_list=(1e-4,),
  #       nb_loss_scale_list=(2.0, 5.0),
  #       b_loss_scale_list=(0.25, 0.5, 1.0),
  #       time_rqmt=5,
  #     ):
  #       recog.center_window_returnn_frame_wise_beam_search(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=checkpoint,
  #       )

  # -------------------------- from-scratch fixed-path training (similar to Atanas) --------------------------------

  for label_decoder_state in ("nb-2linear-ctx1",):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
            win_size_list=(1,),
            label_decoder_state=label_decoder_state,
            use_weight_feedback=False,
            use_att_ctx_in_state=False
    ):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(600,),
        chunked_data_len=95_440,  # -> 600ms -> 100 encoder frames
        nb_loss_scale=6.0,
        use_mgpu=False,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for label_decoder_state in ("nb-2linear-ctx1",):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
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
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
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

  # -------------------------- from-scratch Viterbi training --------------------------------

  for label_decoder_state in ("nb-2linear-ctx1",):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
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
        do_realignments=True,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

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
