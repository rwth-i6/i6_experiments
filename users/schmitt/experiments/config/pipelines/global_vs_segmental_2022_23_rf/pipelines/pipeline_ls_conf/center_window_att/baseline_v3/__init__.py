from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # ------------------------------------- ctx-1 models -------------------------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), label_decoder_state="nb-linear1", blank_decoder_version=5
  ):
    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
        import_model_name=import_model_name,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), label_decoder_state="nb-linear1", blank_decoder_version=3
  ):
    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
        import_model_name=import_model_name,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(5,),
          label_decoder_state="nb-linear1",
          blank_decoder_version=5,
          use_weight_feedback=False,
          use_att_ctx_in_state=False
  ):
    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
        import_model_name=import_model_name,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  # ------------------------------------- blank decoder variants -------------------------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), blank_decoder_version=5,
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
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), blank_decoder_version=6,
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
      )

  # ------------------------------------- from-scratch Viterbi training -------------------------------------

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  # ------------------------------------- best models: KEEP! -------------------------------------

  for use_weight_feedback in (True, False):
    for use_att_ctx_in_state in (True, False):
      for model_alias, config_builder in baseline.center_window_att_baseline_rf(
        win_size_list=(5,), use_weight_feedback=use_weight_feedback, use_att_ctx_in_state=use_att_ctx_in_state
      ):
        for import_model_name in ("glob.conformer.mohammad.5.4",):
          for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(100,),
            const_lr_list=(1e-4,),
            import_model_name=import_model_name,
          ):
            recog.center_window_returnn_frame_wise_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=checkpoint,
            )
            if "129" not in model_alias:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=train_alias,
                config_builder=config_builder,
                checkpoint=checkpoint,
                checkpoint_aliases=("last",),
                # lm_type="trafo",
                # lm_scale_list=(0.4,),
              )
              if use_weight_feedback and use_att_ctx_in_state:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=train_alias,
                  config_builder=config_builder,
                  checkpoint=checkpoint,
                  checkpoint_aliases=("last",),
                  lm_type="trafo",
                  lm_scale_list=(0.54,),
                  ilm_type="mini_att",
                  ilm_scale_list=(0.4,),
                )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
          win_size_list=(129,),
  ):
    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs_list=(100,),
              const_lr_list=(1e-4,),
              import_model_name=import_model_name,
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )
        if "129" not in model_alias:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            # lm_type="trafo",
            # lm_scale_list=(0.4,),
          )
