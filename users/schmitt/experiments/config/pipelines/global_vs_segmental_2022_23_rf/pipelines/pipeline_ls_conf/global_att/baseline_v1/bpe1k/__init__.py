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
    for train_alias, checkpoint in (
            (f"{model_alias}/import_glob.conformer.luca.bpe1k.w-ctc", external_checkpoints["luca-aed-bpe1k-w-ctc"]),
            (f"{model_alias}/import_glob.conformer.luca.bpe1k.wo-ctc", external_checkpoints["luca-aed-bpe1k-wo-ctc"]),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-luca",),
        run_analysis=True,
        analyze_gradients=True,
        plot_att_weights=False,
      )

    for train_alias, checkpoint in train.train_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      time_rqmt=10
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch % 20 == 0 or epoch in (5, 10, 20, 30) or epoch in range(1, 60, 1):
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
          )

    for train_alias, checkpoint in train.train_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
      time_rqmt=80,
      use_mgpu=False,
      batch_size=10_000,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
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
        if epoch % 20 == 0 and epoch not in (160, 20, 40, 360, 400, 440, 240, 100):
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
        if epoch % 20 == 0 and epoch not in (240, 360, 140, 160):
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
          )




