from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          blank_decoder_version=7,
          blank_decoder_opts={"version": 7, "distance_embed_dim": 64, "max_distance": 10}
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

  # ------------------- test importance of past att dep. for different win-sizes ---------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(1, 5, 11, 25,), use_weight_feedback=False, use_att_ctx_in_state=False,
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
        use_recombination="sum",
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
        beam_size_list=(84,),
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(129,), use_weight_feedback=False, use_att_ctx_in_state=False,
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
        use_recombination="sum",
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        lm_type="trafo",
        lm_scale_list=(0.54,),
        ilm_type="mini_att",
        ilm_scale_list=(0.4,),
        subtract_ilm_eos_score=True,
        use_recombination="sum",
        corpus_keys=("dev-other", "test-other"),
        beam_size_list=(12,),
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(1, 11, 25,), use_weight_feedback=False, use_att_ctx_in_state=False,
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
        checkpoint_aliases=("last",),
        lm_type="trafo",
        lm_scale_list=(0.5,),
        ilm_type="mini_att",
        ilm_scale_list=(0.32, 0.4, 0.48),
        subtract_ilm_eos_score=True,
        use_recombination="sum",
        corpus_keys=("dev-other",),
        beam_size_list=(12,),
      )


  # ------------------- test blank decoder v4 (full label ctx) ---------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
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
        use_recombination="sum",
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        lm_type="trafo",
        lm_scale_list=(0.54,),
        ilm_type="mini_att",
        ilm_scale_list=(0.44,),
        use_recombination="sum",
        corpus_keys=("dev-other", "test-other"),
        beam_size_list=(84,),
        subtract_ilm_eos_score=True,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
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
        checkpoint_aliases=("last",),
        run_analysis=True,
        att_weight_seq_tags=[
          "dev-other/116-288045-0017/116-288045-0017",
          "dev-other/116-288045-0014/116-288045-0014",
        ]
      )

  # ------------------------------------- from-scratch Viterbi training -------------------------------------

  # for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #   win_size_list=(5,),
  # ):
  #   for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
  #     alias=model_alias,
  #     config_builder=config_builder,
  #     n_epochs_list=(500,),
  #   ):
  #     recog.center_window_returnn_frame_wise_beam_search(
  #       alias=train_alias,
  #       config_builder=config_builder,
  #       checkpoint=checkpoint,
  #     )

  # ------------------------------------- best models: KEEP! -------------------------------------

  # for win_size in (1, 5, 11, 25, 129):
  #   for model_alias, config_builder in baseline.center_window_att_baseline_rf(
  #           win_size_list=(win_size,), use_weight_feedback=win_size != 1
  #   ):
  #     for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
  #             alias=model_alias,
  #             config_builder=config_builder,
  #             n_epochs_list=(300,),
  #             const_lr_list=(1e-4,),
  #     ):
  #       recog.center_window_returnn_frame_wise_beam_search(
  #         alias=train_alias,
  #         config_builder=config_builder,
  #         checkpoint=checkpoint,
  #       )

  for win_size in (1, 5, 11, 25, 129):
    for model_alias, config_builder in baseline.center_window_att_baseline_rf(
            win_size_list=(win_size,), use_weight_feedback=not win_size == 1,
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
          beam_size_list=(84,),
        )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
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
        checkpoint_aliases=("last",),
        run_analysis=True,
      )

    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
            optimizer_opts={
              "class": "adamw",
              "epsilon": 1e-16,
              "weight_decay": 0.01,
              "weight_decay_modules_blacklist": [
                  "rf.Embedding",
                  "rf.LearnedRelativePositionalEncoding",
              ],
            },
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
  ):
    for num_iterations in (1, 2, 3):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(300,),
              const_lr_list=(1e-4,),
              alignment_augmentation_opts={"max_shift": 1, "num_iterations": num_iterations},
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
        )
