from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name


def run_exps():
  for use_weight_feedback in (True,):
    for model_alias, config_builder in baseline.global_att_baseline_rf(use_weight_feedback=use_weight_feedback):
      for train_alias, checkpoint in (
              (f"{model_alias}/import-global-tf_no-finetuning", external_checkpoints[default_import_model_name]),
      ):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best-4-avg",),
        )

      for train_alias, checkpoint in train.train_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(10, 100),
        const_lr_list=(1e-4,),
      ):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for use_weight_feedback in (True,):
    for model_alias, config_builder in baseline.global_att_baseline_rf(
      use_weight_feedback=use_weight_feedback,
      use_att_ctx_in_state=False,
    ):
      for train_alias, checkpoint in train.train_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
      ):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for use_weight_feedback in (False,):
    for model_alias, config_builder in baseline.global_att_baseline_rf(
      use_weight_feedback=use_weight_feedback,
      use_att_ctx_in_state=False,
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
