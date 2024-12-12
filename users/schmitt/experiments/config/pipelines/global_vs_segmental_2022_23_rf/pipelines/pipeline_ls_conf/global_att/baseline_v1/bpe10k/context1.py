from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import \
  external_checkpoints, default_import_model_name


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(decoder_state="nb-2linear-ctx1"):
    for train_alias, checkpoint in train.train_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      for lm_scale, ilm_scale, beam_size in [
        (0.54, 0.4, 12),
        (0.54, 0.4, 84),
        (0.5, 0.4, 12),
        (0.52, 0.4, 12),
      ]:
        recog.global_att_returnn_label_sync_beam_search(
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
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
        analyze_gradients=True,
      )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          decoder_state="nb-2linear-ctx1",
          use_weight_feedback=False,
          use_att_ctx_in_state=False
  ):
    for train_alias, checkpoint in train.train_import_global_tf(
            alias=model_alias,
            config_builder=config_builder,
            n_epochs_list=(300,),
            const_lr_list=(1e-4,),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

      for lm_scale, ilm_scale, beam_size in [
        (0.54, 0.4, 12),
        # (0.54, 0.4, 84),
        (0.5, 0.4, 12),
        (0.52, 0.4, 12),
      ]:
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          lm_type="trafo",
          lm_scale_list=(lm_scale,),
          ilm_scale_list=(ilm_scale,),
          ilm_type="mini_att",
          beam_size_list=(beam_size,),
          batch_size=10_000,
        )

      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
        analyze_gradients=True,
      )




