from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import data
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import plot_gradient_wrt_enc11
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import lm_checkpoints

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT


def run_exps():
  # ------------------------------ Transducer ------------------------------

  data.analyze_gradients_jobs["baseline_v3_two-stage"] = {"fixed-path": {}, "full-sum": {}}

  n_full_epochs_fixed_path = 30
  n_full_epochs_full_sum = 15
  for (
    alias,
    win_size,
    use_current_frame_in_readout_w_double_gate,
    do_full_sum,
    use_current_frame_in_readout_w_gate,
    use_current_frame_in_readout_w_gate_v,
    use_att_ctx_in_state,
    use_sep_att_encoder,
    use_sep_h_t_readout,
  ) in [
    ("v1", 1, False, True, False, 1, False, False, False),  # standard transducer
    ("v2", None, False, True, False, 1, False, False, False),  # transducer with LSTM attention w/o att ctx in state
    ("v3", None, False, False, False, 1, True, False, False),  # transducer with LSTM attention w/ att ctx in state
    ("v4", None, True, False, False, 1, False, False, False),  # transducer with LSTM attention and double gate
    ("v5", None, True, False, False, 1, True, False, False),  # transducer with LSTM attention and double gate and att ctx in state
    ("v6", None, False, False, True, 2, False, False, False),  # transducer with LSTM attention and single gate
    ("v7", None, False, False, False, 1, True, True, False),  # transducer with LSTM attention w/ att ctx in state w/ sep att encoder
    ("v8", None, False, False, False, 1, True, False, True),  # transducer with LSTM attention w/ att ctx in state w/ sep h_t readout
  ]:
    gpu_mem_rqmts = [24]
    if alias == "v1":
      gpu_mem_rqmts.append(11)
    if alias in ("v4", "v5", "v6", "v7", "v8"):
      gpu_mem_rqmts = [11]

    for gpu_mem_rqmt in gpu_mem_rqmts:
      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 2
        batch_size = 30_000
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20
        n_epochs_full_sum = n_full_epochs_full_sum * 20
      else:
        use_mgpu = True
        accum_grad_multiple_step = 4
        if alias in ("v4", "v5"):
          batch_size = 12_000
        elif alias in ("v6",):
          batch_size = 10_000
        elif alias in ("v7",):
          batch_size = 8_000
        else:
          batch_size = 15_000
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20 // 4
        n_epochs_full_sum = n_full_epochs_full_sum * 20 // 4

      for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
              win_size_list=(win_size,),
              blank_decoder_version=4,
              use_att_ctx_in_state=use_att_ctx_in_state,
              use_weight_feedback=False,
              bpe_vocab_size=1056,
              use_correct_dim_tags=True,
              behavior_version=21,
              use_current_frame_in_readout=win_size is None and not use_current_frame_in_readout_w_double_gate and not use_current_frame_in_readout_w_gate and not use_sep_h_t_readout,
              use_current_frame_in_readout_w_double_gate=use_current_frame_in_readout_w_double_gate,
              use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
              use_current_frame_in_readout_w_gate_v=use_current_frame_in_readout_w_gate_v,
              use_sep_att_encoder=use_sep_att_encoder,
              use_sep_h_t_readout=use_sep_h_t_readout,
      ):
        keep_epochs_step_fixed_path = n_epochs_fixed_path // 10
        keep_epochs_fixed_path = list(range(keep_epochs_step_fixed_path, n_epochs_fixed_path, keep_epochs_step_fixed_path))
        for fixed_path_train_alias, fixed_path_checkpoint in train.train_center_window_att(
                alias=model_alias,
                config_builder=config_builder,
                n_epochs=n_epochs_fixed_path,
                batch_size=batch_size,
                use_mgpu=use_mgpu,
                use_speed_pert=False,
                training_type="fixed-path",
                hdf_targets=LibrispeechBPE1056_CTC_ALIGNMENT.alignment_paths,
                keep_epochs=keep_epochs_fixed_path,
                filter_data_len=19.5 * 16_000,
                ctc_aux_loss_layers=(4, 8),
                gpu_mem_rqmt=gpu_mem_rqmt,
                accum_grad_multiple_step=accum_grad_multiple_step,
                lr_scheduling_opts={
                  "type": "dyn_lr_piecewise_linear_epoch-wise_v2",
                  "init_lr": 1e-5,
                  "peak_lr": 1e-3,
                  "lr2": 1e-5,
                },
        ):
          recog.center_window_returnn_frame_wise_beam_search(
            alias=fixed_path_train_alias,
            config_builder=config_builder,
            checkpoint=fixed_path_checkpoint,
          )

          if alias == "v1" and gpu_mem_rqmt == 24:
            for lm_scale, ilm_scale in [
              (0.54, 0.4),
              (0.6, 0.4),
              (0.7, 0.4),
              (0.7, 0.3),
              (0.8, 0.4),
            ]:
              lm_alias = "1k_max-seq-length-112_24-layers_512-dim"
              recog.center_window_returnn_frame_wise_beam_search(
                alias=fixed_path_train_alias,
                config_builder=config_builder,
                checkpoint=fixed_path_checkpoint,
                checkpoint_aliases=("last",),
                lm_type="trafo",
                lm_scale_list=(lm_scale,),
                ilm_scale_list=(ilm_scale,),
                ilm_type="mini_att",
                lm_alias=lm_alias,
                lm_checkpoint=lm_checkpoints[lm_alias],
              )

          pipeline = recog.center_window_returnn_frame_wise_beam_search(
            alias=fixed_path_train_alias,
            config_builder=config_builder,
            checkpoint=fixed_path_checkpoint,
            checkpoint_aliases=("last",),
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
          data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"][alias] = pipeline.decoding_exps[0].analyze_gradients_job

          for epoch, chckpt in fixed_path_checkpoint["checkpoints"].items():
            if epoch in keep_epochs_fixed_path:
              separate_readout_alphas = [None]
              if alias == "v8":
                separate_readout_alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
              for separate_readout_alpha in separate_readout_alphas:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=fixed_path_train_alias,
                  config_builder=config_builder,
                  checkpoint=chckpt,
                  checkpoint_aliases=(f"epoch-{epoch}",),
                  separate_readout_alpha=separate_readout_alpha,
                )
              if win_size is None and epoch in [45, 60, 150, 180, 240, 600]:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=fixed_path_train_alias,
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

        if not do_full_sum:
          continue
        keep_epochs_step_full_sum = n_epochs_full_sum // 10
        keep_epochs_full_sum = list(range(keep_epochs_step_full_sum, n_epochs_full_sum, keep_epochs_step_full_sum))
        peak_lrs = []
        if gpu_mem_rqmt == 24 and alias == "v1":
          peak_lrs.append(3e-4)
        for peak_lr in peak_lrs:
          for full_sum_train_alias, full_sum_checkpoint in train.train_center_window_att(
                  alias=model_alias,
                  config_builder=config_builder,
                  n_epochs=n_epochs_full_sum,
                  batch_size=batch_size // 3,
                  use_mgpu=use_mgpu,
                  use_speed_pert=True,
                  training_type="full-sum",
                  keep_epochs=keep_epochs_full_sum,
                  filter_data_len=19.5 * 16_000,
                  ctc_aux_loss_layers=(4, 8),
                  gpu_mem_rqmt=gpu_mem_rqmt,
                  accum_grad_multiple_step=accum_grad_multiple_step * 3,
                  checkpoint_path=fixed_path_checkpoint["checkpoints"][n_epochs_fixed_path],
                  checkpoint_alias="fixed-path_30-full-epochs",
                  lr_scheduling_opts={
                    "type": "dyn_lr_piecewise_linear_epoch-wise_v2",
                    "init_lr": peak_lr,
                    "peak_lr": peak_lr,
                    "lr2": peak_lr / 5,
                  },
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

  plot_gradient_wrt_enc11(
    data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v2"],
    alias=f"{base_alias}/gradients_lstm-att_wo-att-ctx-in-state_fixed-path_epoch-600"
  )
