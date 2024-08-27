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

    # v1: training, where i observed the flipped encoder after about 60 sub-epochs
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
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

    # v2: same as v1, but use epoch-wise OCLR
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      keep_epochs=list(range(1, 240)) + [500],
      lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in [22, 55]:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

    # v3: same as v2, but filter out data > 19.5s
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      keep_epochs=list(range(1, 240)) + [500],
      lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
      filter_data_len=19.5 * 16_000,  # sample rate 16kHz
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in [31, 55]:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

    # v4: same as v2, but filter out targets > 75
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      keep_epochs=list(range(1, 240)) + [500],
      lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
      filter_target_len=75,  # sample rate 16kHz
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch == 116:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

    # v2_big: same as v2, but on 24gb GPU with batch size 40k
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=2_000,
      batch_size=35_000,
      keep_epochs=list(range(1, 240)) + [2_000],
      lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
      gpu_mem_rqmt=24,
      cluster_reservation_string="hlt_12",
      accum_grad_multiple_step=2,
      use_mgpu=False,
      use_torch_amp=False,
    ):
      pass
      # for epoch, chckpt in checkpoint["checkpoints"].items():
      #   recog.global_att_returnn_label_sync_beam_search(
      #     alias=train_alias,
      #     config_builder=config_builder,
      #     checkpoint=chckpt,
      #     checkpoint_aliases=(f"epoch-{epoch}",),
      #     run_analysis=True,
      #     analyze_gradients=True,
      #     only_do_analysis=True,
      #   )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=True,
          label_type="bpe1056",
          disable_enc_self_att_until_epoch=21,
  ):
    # v5_big: same as v2_big, but enable self attention only after 20 sub-epochs (1 full epoch)
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=2_000,
      batch_size=35_000,
      keep_epochs=list(range(1, 240)) + [2_000],
      lr_scheduling_type="dyn_lr_piecewise_linear_epoch-wise_v2",
      gpu_mem_rqmt=24,
      cluster_reservation_string="hlt_12",
      accum_grad_multiple_step=2,
      use_mgpu=False,
      use_torch_amp=False,
    ):
      pass
      # for epoch, chckpt in checkpoint["checkpoints"].items():
      #   recog.global_att_returnn_label_sync_beam_search(
      #     alias=train_alias,
      #     config_builder=config_builder,
      #     checkpoint=chckpt,
      #     checkpoint_aliases=(f"epoch-{epoch}",),
      #     run_analysis=True,
      #     analyze_gradients=True,
      #     only_do_analysis=True,
      #   )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=True,
          decoder_state="nb-2linear-ctx1",
          label_type="bpe1056",
  ):
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
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
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
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




