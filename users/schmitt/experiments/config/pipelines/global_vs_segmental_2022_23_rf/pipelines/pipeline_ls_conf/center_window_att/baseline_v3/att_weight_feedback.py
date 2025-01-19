from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
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

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
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

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
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

  for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
          win_size_list=(1,), use_weight_feedback=False, use_att_ctx_in_state=False,
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
        analyze_gradients=True,
      )

  for use_weight_feedback, use_att_ctx_in_state in [(True, False), (False, True), (False, False)]:
    for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
            win_size_list=(5,), use_weight_feedback=use_weight_feedback, use_att_ctx_in_state=use_att_ctx_in_state,
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

        for lm_scale, ilm_scale, beam_size in [
          (0.54, 0.4, 12),
          (0.54, 0.4, 84),
        ]:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            lm_type="trafo",
            lm_scale_list=(lm_scale,),
            ilm_scale_list=(ilm_scale,),
            ilm_type="mini_att",
            beam_size_list=(beam_size,),
          )
