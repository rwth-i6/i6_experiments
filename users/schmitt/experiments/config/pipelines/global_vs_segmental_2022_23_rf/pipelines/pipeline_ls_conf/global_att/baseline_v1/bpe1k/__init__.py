from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=True,
          label_type="bpe1056",
  ):
    # for train_alias, checkpoint in (
    #         (f"{model_alias}/import_glob.conformer.luca.bpe1k", external_checkpoints["luca-aed-bpe1k"]),
    # ):
    #   recog.global_att_returnn_label_sync_beam_search(
    #     alias=train_alias,
    #     config_builder=config_builder,
    #     checkpoint=checkpoint,
    #     checkpoint_aliases=("best-luca",),
    #     run_analysis=True,
    #     analyze_gradients=True,
    #     plot_att_weights=False,
    #   )
    #
    # for train_alias, checkpoint in train.train_import_global_tf(
    #   alias=model_alias,
    #   config_builder=config_builder,
    #   n_epochs_list=(300,),
    #   const_lr_list=(1e-4,),
    #   import_model_name="luca-aed-bpe1k",
    # ):
    #   recog.global_att_returnn_label_sync_beam_search(
    #     alias=train_alias,
    #     config_builder=config_builder,
    #     checkpoint=checkpoint,
    #   )

    for train_alias, checkpoint in train.train_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
          )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=True,
          decoder_state="nb-2linear-ctx1",
          label_type="bpe1056",
  ):
    for train_alias, checkpoint in train.train_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      use_mgpu=False,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
          )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=False,
          decoder_state="nb-2linear-ctx1",
          label_type="bpe1056",
  ):
    for train_alias, checkpoint in train.train_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      use_mgpu=False,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 5 == 0:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
          )




