from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import \
  external_checkpoints, default_import_model_name


def run_exps():
  for model_alias, config_builder in baseline.global_att_baseline_rf(use_weight_feedback=True):
    # v5: same as v3, but use bpe size 10k
    for random_seed, gpu_mem_rqmt, ctc_aux_loss_layers in [
      [None, 11, None],
      [1234, 11, None],
      [None, 24, (4, 8)],
    ]:

      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 2
        batch_size = 35_000
        n_epochs = 2_000
      else:
        use_mgpu = True
        accum_grad_multiple_step = 4
        batch_size = 15_000
        n_epochs = 500

      for train_alias, checkpoint in train.train_global_att(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs=n_epochs,
        filter_data_len=19.5 * 16_000,  # sample rate 16kHz
        random_seed=random_seed,
        use_mgpu=use_mgpu,
        accum_grad_multiple_step=accum_grad_multiple_step,
        ctc_aux_loss_layers=ctc_aux_loss_layers,
        batch_size=batch_size,
        gpu_mem_rqmt=gpu_mem_rqmt,
      ):
        if random_seed != 1234:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            corpus_keys=("dev-other", "dev-clean", "test-other", "test-clean"),
          )
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            run_analysis=True,
            analysis_dump_gradients=True,
            only_do_analysis=True,
            corpus_keys=("train",),
            att_weight_seq_tags=None,
          )
        for epoch, chckpt in checkpoint["checkpoints"].items():
          if epoch == 70 or epoch in range(1, 60, 5):
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              run_analysis=True,
              analyze_gradients=True,
              only_do_analysis=True,
            )
          if epoch in range(10, 70, 10):
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              run_analysis=True,
              only_do_analysis=True,
              analyze_gradients=True,
              analsis_analyze_gradients_plot_log_gradients=False,
              analysis_analyze_gradients_plot_encoder_layers=True,
              att_weight_seq_tags=[
                "train-other-960/1246-124548-0042/1246-124548-0042",
                "train-other-960/40-222-0033/40-222-0033",
                "train-other-960/103-1240-0038/103-1240-0038",
              ],
              corpus_keys=("train",),
            )

    for train_alias, checkpoint in (
            (f"{model_alias}/import_{default_import_model_name}", external_checkpoints[default_import_model_name]),
            (f"{model_alias}/import_glob.conformer.mohammad.5.4", external_checkpoints["glob.conformer.mohammad.5.4"]),
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        run_analysis=True,
        analyze_gradients=True,
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        run_analysis=True,
        only_do_analysis=True,
        analyze_gradients=True,
        analsis_analyze_gradients_plot_log_gradients=False,
        analysis_analyze_gradients_plot_encoder_layers=False,
        att_weight_seq_tags=[
          "train-other-960/1246-124548-0042/1246-124548-0042",
          "train-other-960/40-222-0033/40-222-0033",
          "train-other-960/103-1240-0038/103-1240-0038",
        ],
        corpus_keys=("train",),
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        run_analysis=True,
        analysis_dump_gradients=True,
        only_do_analysis=True,
        corpus_keys=("train",),
        att_weight_seq_tags=None,
      )
      for concat_num in (8,):
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best-4-avg",),
          batch_size=30_000,
          run_analysis=True,
          analyze_gradients=True,
          plot_att_weights=False,
          concat_num=concat_num,
          att_weight_seq_tags=[
            "dev-other/116-288045-0008/116-288045-0008;dev-other/116-288045-0009/116-288045-0009;dev-other/116-288045-0010/116-288045-0010;dev-other/116-288045-0011/116-288045-0011;dev-other/116-288045-0012/116-288045-0012;dev-other/116-288045-0013/116-288045-0013;dev-other/116-288045-0014/116-288045-0014;dev-other/116-288045-0015/116-288045-0015",
          ]
        )
      for corpus_key in [
        "dev-other_0.1-5.1",
        "dev-other_5.1-10.1",
        "dev-other_10.1-15.1",
        "dev-other_15.1-20.1",
        "dev-other_20.1-25.1",
        "dev-other_25.1-30.1",
        "dev-other_30.1-35.1",
      ]:
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best-4-avg",),
          corpus_keys=(corpus_key,),
        )

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
        checkpoint_aliases=("last",),
        run_analysis=True,
        analyze_gradients=True,
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        beam_size_list=(100,),
        corpus_keys=("test-other", "dev-other"),
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        corpus_keys=("test-other",),
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        lm_type="trafo",
        lm_scale_list=(0.5, 0.52, 0.54,),
        ilm_scale_list=(0.4,),
        ilm_type="mini_att",
        beam_size_list=(12, 84),
        # corpus_keys=("dev-other", "test-other"),
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        lm_type="trafo",
        lm_scale_list=(0.54,),
        ilm_scale_list=(0.4,),
        ilm_type="mini_att",
        beam_size_list=(84,),
        corpus_keys=("test-other",),
      )

      for concat_num in (2, 4, 8, 10, 20, 30):
        if concat_num == 20:
          batch_size = 23_000
        elif concat_num == 30:
          batch_size = 30_000
        elif concat_num == 100:
          batch_size = 61_000
        else:
          batch_size = 15_000
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last",),
          batch_size=batch_size,
          concat_num=concat_num,
        )

      for lm_scale, ilm_scale, beam_size, length_normalization_exponent in [
        (0.0, 0.0, 12, 0.0),
        (0.0, 0.0, 84, 0.0),
        (0.0, 0.0, 84, 1.0),
        (0.54, 0.4, 12, 0.0),
        (0.54, 0.4, 84, 0.0),
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
          length_normalization_exponent=length_normalization_exponent,
        )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=False, use_att_ctx_in_state=False
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
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("best-4-avg",),
        run_analysis=True,
        analyze_gradients=True,
        att_weight_seq_tags=[
          "dev-other/4572-112375-0006/4572-112375-0006",
          "dev-other/116-288048-0025/116-288048-0025",
          "dev-other/1701-141760-0002/1701-141760-0002",
          "dev-other/1630-102884-0005/1630-102884-0005",
          "dev-other/7697-105815-0051/7697-105815-0051",
        ]
      )
      # recog.global_att_returnn_label_sync_beam_search(
      #   alias=train_alias,
      #   config_builder=config_builder,
      #   checkpoint=checkpoint,
      #   checkpoint_aliases=("best-4-avg",),
      #   run_analysis=True,
      #   analyze_gradients=True,
      #   att_weight_seq_tags=[
      #     "train-other-960/1246-124548-0042/1246-124548-0042",
      #     "train-other-960/40-222-0033/40-222-0033",
      #     "train-other-960/103-1240-0038/103-1240-0038",
      #   ],
      #   corpus_keys=("train-sample",)
      # )
      for lm_scale, ilm_scale, beam_size in [
        (0.2, 0.1, 12),
        (0.1, 0.1, 12),
        (0.54, 0.4, 12),
        (0.3, 0.1, 12),
        (0.3, 0.2, 12),
        (0.4, 0.2, 12),
        (0.5, 0.3, 12),
      ]:
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best-4-avg",),
          lm_type="trafo",
          lm_scale_list=(lm_scale,),
          ilm_scale_list=(ilm_scale,),
          ilm_type="mini_att",
          beam_size_list=(beam_size,),
          batch_size=10_000,
        )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=False, use_att_ctx_in_state=True
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

      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        run_analysis=True,
        analyze_gradients=True,
        att_weight_seq_tags=[
          "dev-other/4572-112375-0006/4572-112375-0006",
          "dev-other/116-288048-0025/116-288048-0025",
          "dev-other/1701-141760-0002/1701-141760-0002",
          "dev-other/1630-102884-0005/1630-102884-0005",
          "dev-other/7697-105815-0051/7697-105815-0051",
        ]
      )

      for lm_scale, ilm_scale, beam_size, length_normalization_exponent in [
        (0.54, 0.4, 12, 1.0),
        (0.54, 0.4, 84, 1.0),
        (0.54, 0.4, 12, 0.0),
        (0.54, 0.4, 84, 0.0),
        (0.0, 0.0, 84, 0.0),
        (0.0, 0.0, 12, 0.0),
        (0.0, 0.0, 84, 1.0),
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
          length_normalization_exponent=length_normalization_exponent,
        )

  for model_alias, config_builder in baseline.global_att_baseline_rf(
          use_weight_feedback=True, use_att_ctx_in_state=False
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
        checkpoint_aliases=("best-4-avg",),
        run_analysis=True,
        analyze_gradients=True,
      )

      for lm_scale, ilm_scale, beam_size in [
        (0.2, 0.1, 12),
        (0.1, 0.1, 12),
        (0.54, 0.4, 12),
        (0.3, 0.1, 12),
        (0.3, 0.2, 12),
        (0.4, 0.2, 12),
        (0.5, 0.3, 12),
      ]:
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("best-4-avg",),
          lm_type="trafo",
          lm_scale_list=(lm_scale,),
          ilm_scale_list=(ilm_scale,),
          ilm_type="mini_att",
          beam_size_list=(beam_size,),
        )
