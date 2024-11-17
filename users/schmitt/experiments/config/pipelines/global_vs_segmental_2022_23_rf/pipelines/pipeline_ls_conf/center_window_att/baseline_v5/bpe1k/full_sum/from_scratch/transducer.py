from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import data
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import plot_gradient_wrt_enc11
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT


def run_exps():
  # ------------------------------ Transducer ------------------------------

  data.analyze_gradients_jobs["baseline_v5_full-sum"] = {}
  n_full_epochs_full_sum = 45
  for (
    alias,
    win_size,
    use_trafo_att,
    use_trafo_att_wo_cross_att,
  ) in [
    ("v1", None, True, False),  # transducer with trafo att
    ("v2", None, True, True),  # transducer with trafo self att but no cross-att
  ]:
    data.analyze_gradients_jobs["baseline_v5_full-sum"][alias] = {}

    gpu_mem_rqmts = [24]

    for gpu_mem_rqmt in gpu_mem_rqmts:
      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 4
        batch_size = 12_000
        n_epochs_full_sum = n_full_epochs_full_sum * 20
      else:
        use_mgpu = True
        accum_grad_multiple_step = 12
        batch_size = 5_000
        n_epochs_full_sum = n_full_epochs_full_sum * 20 // 4

      for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
              win_size_list=(win_size,),
              blank_decoder_version=4,
              use_att_ctx_in_state=False,
              use_weight_feedback=False,
              bpe_vocab_size=1056,
              use_correct_dim_tags=True,
              behavior_version=21,
              use_trafo_att=use_trafo_att,
              use_current_frame_in_readout=win_size is None,
              use_trafo_att_wo_cross_att=use_trafo_att_wo_cross_att,
      ):
        keep_epochs_step_full_sum = n_epochs_full_sum // 10
        keep_epochs_full_sum = list(range(keep_epochs_step_full_sum, n_epochs_full_sum, keep_epochs_step_full_sum))
        for full_sum_train_alias, full_sum_checkpoint in train.train_center_window_att(
                alias=model_alias,
                config_builder=config_builder,
                n_epochs=n_epochs_full_sum,
                batch_size=batch_size,
                use_mgpu=use_mgpu,
                use_speed_pert=True,
                training_type="full-sum",
                keep_epochs=keep_epochs_full_sum,
                filter_data_len=19.5 * 16_000,
                ctc_aux_loss_layers=(4, 8),
                gpu_mem_rqmt=gpu_mem_rqmt,
                accum_grad_multiple_step=accum_grad_multiple_step,
        ):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=full_sum_train_alias,
            config_builder=config_builder,
            checkpoint=full_sum_checkpoint,
          )

          for epoch, chckpt in full_sum_checkpoint["checkpoints"].items():
            if epoch in keep_epochs_full_sum:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=full_sum_train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
              )

              pipeline = recog.center_window_returnn_frame_wise_beam_search(
                alias=full_sum_train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
                run_analysis=True,
                only_do_analysis=True,
                analyze_gradients=True,
                analsis_analyze_gradients_plot_log_gradients=True,
                analysis_analyze_gradients_plot_encoder_layers=True,
                analysis_ground_truth_hdf=LibrispeechBPE1056_CTC_ALIGNMENT.alignment_paths["train"],
                att_weight_seq_tags=[
                  "train-other-960/1246-124548-0042/1246-124548-0042",
                  "train-other-960/40-222-0033/40-222-0033",
                  "train-other-960/103-1240-0038/103-1240-0038",
                ],
                corpus_keys=("train",),
              )
              data.analyze_gradients_jobs["baseline_v5_full-sum"][alias][f"epoch-{epoch}"] = pipeline.decoding_exps[0].analyze_gradients_job

  plot_gradient_wrt_enc11(
    data.analyze_gradients_jobs["baseline_v5_full-sum"]["v1"][f"epoch-360"],
    alias=f"{base_alias}/gradients_trafo-att_full-sum_epoch-360",
  )
