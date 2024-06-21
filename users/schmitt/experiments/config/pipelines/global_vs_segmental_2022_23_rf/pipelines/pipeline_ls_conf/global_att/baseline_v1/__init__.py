from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(use_weight_feedback=True):
    for train_alias, checkpoint in (
            (f"{model_alias}/import_{default_import_model_name}", external_checkpoints[default_import_model_name]),
            (f"{model_alias}/import_glob.conformer.mohammad.5.4", external_checkpoints["glob.conformer.mohammad.5.4"]),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
      )

    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
        import_model_name=import_model_name,
      ):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best",),
          lm_type="trafo",
          lm_scale_list=(0.4, 0.54),
          ilm_scale_list=(0.4,),
          ilm_type="mini_att",
          beam_size_list=(12,)
        )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=False, use_att_ctx_in_state=False
  ):
    for import_model_name in ("glob.conformer.mohammad.5.4",):
      for train_alias, checkpoint in train.train_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
        import_model_name=import_model_name,
      ):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )



