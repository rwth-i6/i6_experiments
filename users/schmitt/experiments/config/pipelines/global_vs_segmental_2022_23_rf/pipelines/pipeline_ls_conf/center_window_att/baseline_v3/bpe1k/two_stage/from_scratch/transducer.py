from i6_core.returnn import PtCheckpoint
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

from sisyphus import Path

def run_exps():
  # ------------------------------ Transducer ------------------------------

  data.analyze_gradients_jobs["baseline_v3_two-stage"] = {"fixed-path": {}, "full-sum": {}}

  n_full_epochs_fixed_path = 30
  n_full_epochs_full_sum = 15
  for (
    alias,
    win_size,
    use_current_frame_in_readout,
    use_current_frame_in_readout_w_double_gate,
    do_full_sum,
    use_current_frame_in_readout_w_gate,
    use_current_frame_in_readout_w_gate_v,
    use_att_ctx_in_state,
    use_sep_att_encoder,
    use_sep_h_t_readout,
    use_weight_feedback,
    mask_att_around_h_t,
    n_full_epochs_fixed_path,
    n_full_epochs_full_sum,
    batch_size11,
    batch_size24,
    window_step_size,
    random_seed,
  ) in [
    ("v1", 1, False, False, True, False, 1, False, False, False, False, False, 30, 15, 15_000, 30_000, 1, None),  # standard transducer
    ("v1_long_two-stage", 1, False, False, True, False, 1, False, False, False, False, False, 60, 40, 15_000, 30_000, 1, None),  # standard transducer
    ("v1_long_two-stage-rand-seed-1234", 1, False, False, True, False, 1, False, False, False, False, False, 60, 40, 15_000, 30_000, 1, 1234),  # standard transducer
    # ("v1_long_fixed-path", 1, False, False, True, False, 1, False, False, False, False, False, 100, 0, 15_000, 30_000, 1),  # standard transducer
    ("v2", None, True, False, True, False, 1, False, False, False, False, False, 30, 15, 12_000, 30_000, 1, None),  # transducer with LSTM attention w/o att ctx in state
    ("v2_long_two-stage", None, True, False, True, False, 1, False, False, False, False, False, 60, 40, 12_000, 30_000, 1, None),  # transducer with LSTM attention w/o att ctx in state
    ("v3", None, True, False, True, False, 1, True, False, False, False, False, 30, 15, 15_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state
    ("v3_long_two-stage", None, True, False, True, False, 1, True, False, False, False, False, 60, 40, 15_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state
    # ("v3_long_fixed-path", None, True, False, False, False, 1, True, False, False, False, False, 100, 0, 15_000, 30_000, 1),  # transducer with LSTM attention w/ att ctx in state
    ("v4", None, False, True, False, False, 1, False, False, False, False, False, 30, 15, 12_000, 30_000, 1, None),  # transducer with LSTM attention and double gate
    ("v4_long_two-stage", None, False, True, False, False, 1, False, False, False, False, False, 60, 40, 12_000, 30_000, 1, None),  # transducer with LSTM attention and double gate
    ("v5", None, False, True, False, False, 1, True, False, False, False, False, 30, 15, 12_000, 30_000, 1, None),  # transducer with LSTM attention and double gate and att ctx in state
    ("v5_long_two-stage", None, False, True, False, False, 1, True, False, False, False, False, 60, 40, 12_000, 30_000, 1, None),  # transducer with LSTM attention and double gate and att ctx in state
    ("v6", None, False, False, False, True, 2, False, False, False, False, False, 30, 15, 10_000, 30_000, 1, None),  # transducer with LSTM attention and single gate
    ("v6_long_two-stage", None, False, False, False, True, 2, False, False, False, False, False, 60, 40, 10_000, 30_000, 1, None),  # transducer with LSTM attention and single gate
    ("v7", None, True, False, False, False, 1, True, True, False, False, False, 30, 15, 8_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state w/ sep att encoder
    ("v8", None, False, False, False, False, 1, True, False, True, False, False, 30, 15, 15_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state w/ sep h_t readout
    ("v8_long_two-stage", None, False, False, True, False, 1, True, False, True, False, False, 60, 40, 15_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state w/ sep h_t readout
    ("v9", None, True, False, False, False, 1, True, False, False, True, False, 30, 15, 15_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state and w/ weight feedback
    ("v9_long_two-stage", None, True, False, False, False, 1, True, False, False, True, False, 60, 30, 15_000, 30_000, 1, None),  # transducer with LSTM attention w/ att ctx in state and w/ weight feedback
    ("v11", None, False, False, False, False, 1, True, False, False, False, False, 30, 15, 15_000, 30_000, 1, None),  # transducer w/o h_t in Readout with LSTM attention w/ att ctx in state
    ("v11_long_two-stage", None, False, False, False, False, 1, True, False, False, False, False, 60, 40, 15_000, 30_000, 1, None),  # transducer w/o h_t in Readout with LSTM attention w/ att ctx in state
    ("v12", None, False, False, False, False, 1, False, False, False, False, False, 30, 15, 12_000, 30_000, 1, None),  # transducer w/o h_t in Readout with LSTM attention w/o att ctx in state
    ("v12_long_two-stage", None, False, False, False, False, 1, False, False, False, False, False, 60, 40, 12_000, 30_000, 1, None),  # transducer w/o h_t in Readout with LSTM attention w/o att ctx in state
    ("v13_long_two-stage", None, False, False, False, False, 1, True, False, False, True, False, 60, 40, 15_000, 30_000, 1, None),  # transducer w/o h_t in Readout with LSTM attention w/ att ctx in state w/ weight feedback
    ("v14_long_two-stage", None, False, True, False, False, 1, True, False, False, True, False, 60, 40, 12_000, 30_000, 1, None),  # transducer with LSTM attention and double gate and att ctx in state
  ]:
    data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"][alias] = {}

    gpu_mem_rqmts = [24]
    if alias in ("v1", "v2", "v3"):
      gpu_mem_rqmts.append(11)
    if alias in ("v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v12"):
      gpu_mem_rqmts = [11]

    if "long" in alias:
      gpu_mem_rqmts = [24]

    for gpu_mem_rqmt in gpu_mem_rqmts:
      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 2
        batch_size = batch_size24
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20
        n_epochs_full_sum = n_full_epochs_full_sum * 20
      else:
        use_mgpu = True
        accum_grad_multiple_step = 4
        batch_size = batch_size11
        # if alias in ("v4", "v5"):
        #   batch_size = 12_000
        # elif alias in ("v6",):
        #   batch_size = 10_000
        # elif alias in ("v7",):
        #   batch_size = 8_000
        # else:
        #   batch_size = 15_000
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20 // 4
        n_epochs_full_sum = n_full_epochs_full_sum * 20 // 4

      for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
              win_size_list=(win_size,),
              blank_decoder_version=4,
              use_att_ctx_in_state=use_att_ctx_in_state,
              use_weight_feedback=use_weight_feedback,
              bpe_vocab_size=1056,
              use_correct_dim_tags=True,
              behavior_version=21,
              # use_current_frame_in_readout=win_size is None and not use_current_frame_in_readout_w_double_gate and not use_current_frame_in_readout_w_gate and not use_sep_h_t_readout,
              use_current_frame_in_readout=use_current_frame_in_readout,
              use_current_frame_in_readout_w_double_gate=use_current_frame_in_readout_w_double_gate,
              use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
              use_current_frame_in_readout_w_gate_v=use_current_frame_in_readout_w_gate_v,
              use_sep_att_encoder=use_sep_att_encoder,
              use_sep_h_t_readout=use_sep_h_t_readout,
              window_step_size=window_step_size,
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
                mask_att_around_h_t=mask_att_around_h_t,
                random_seed=random_seed,
        ):
          checkpoint_aliases = ("last",)
          if alias in ("v4", "v7", "v8", "v11", "v12"):
            checkpoint_aliases = ("last", "best", "best-4-avg")
          if alias in ("v2", "v3", "v9") and gpu_mem_rqmt == 11:
            checkpoint_aliases = ("last", "best", "best-4-avg")
          if "long" in alias:
            checkpoint_aliases = ("last", "best", "best-4-avg")

          if not use_sep_h_t_readout:
            recog.center_window_returnn_frame_wise_beam_search(
              alias=fixed_path_train_alias,
              config_builder=config_builder,
              checkpoint=fixed_path_checkpoint,
              checkpoint_aliases=checkpoint_aliases,
            )
          elif alias == "v8_long_two-stage":
            for att_readout_scale, h_t_readout_scale in [
              (1.0, 0.0),
              (0.0, 1.0),
              (1.0, 0.1),
              (1.0, 0.2),
              (1.0, 0.3),
              (1.0, 0.4),
              (1.0, 0.5),
              (1.0, 0.6),
              (1.0, 0.7),
              (1.0, 0.8),
              (1.0, 0.9),
              (1.0, 1.0),
              (1.0, 1.1),
              (1.0, 1.2),
              (1.0, 1.4),
              (1.0, 1.6),
              (1.0, 1.8),
            ]:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=fixed_path_train_alias,
                config_builder=config_builder,
                checkpoint=fixed_path_checkpoint,
                checkpoint_aliases=("last",),
                att_readout_scale=att_readout_scale,
                h_t_readout_scale=h_t_readout_scale,
              )

          pipeline = recog.center_window_returnn_frame_wise_beam_search(
            alias=fixed_path_train_alias,
            config_builder=config_builder,
            checkpoint=fixed_path_checkpoint,
            checkpoint_aliases=("best-4-avg",) if alias == "v11_long_two-stage" else ("last",),
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
          data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"][alias][f"{gpu_mem_rqmt}gb-gpu"] = pipeline.decoding_exps[0].analyze_gradients_job

          for epoch, chckpt in fixed_path_checkpoint["checkpoints"].items():
            if epoch in keep_epochs_fixed_path:
              if not use_sep_h_t_readout:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=fixed_path_train_alias,
                  config_builder=config_builder,
                  checkpoint=chckpt,
                  checkpoint_aliases=(f"epoch-{epoch}",),
                )
              if (
                      win_size is None and epoch in [30, 45, 60, 150, 180, 240, 600]
              ) or alias == "v14_long_two-stage" and epoch == 1080:
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

        if not do_full_sum or n_full_epochs_full_sum == 0:
          continue
        keep_epochs_step_full_sum = n_epochs_full_sum // 10
        keep_epochs_full_sum = list(range(keep_epochs_step_full_sum, n_epochs_full_sum, keep_epochs_step_full_sum))
        params = []
        if gpu_mem_rqmt == 24 and alias in ("v1", "v2", "v3"):
          params.append((3e-4, False))
        if "long" in alias:
          params.append((1e-4, True))
        for peak_lr, use_normalized_loss in params:
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
                  use_normalized_loss=use_normalized_loss,
                  random_seed=random_seed,
          ):
            if not use_sep_h_t_readout:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=full_sum_train_alias,
                config_builder=config_builder,
                checkpoint=full_sum_checkpoint,
                checkpoint_aliases=("last", "best", "best-4-avg"),
              )
            else:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=full_sum_train_alias,
                config_builder=config_builder,
                checkpoint=full_sum_checkpoint,
                checkpoint_aliases=("best-4-avg",),
                att_readout_scale=1.0,
                h_t_readout_scale=0.5,
                corpus_keys=("test-other",)
              )

            if use_sep_h_t_readout and "long" in alias:
              for att_readout_scale, h_t_readout_scale in [
                (1.0, 0.1),
                (1.0, 0.1),
                (1.0, 0.2),
                (1.0, 0.3),
                (1.0, 0.4),
                (1.0, 0.5),
                (1.0, 0.6),
                (1.0, 0.7),
                (1.0, 0.8),
                (1.0, 0.9),
                (1.0, 1.1),
                (1.0, 1.2),
                (1.0, 1.4),
                (1.0, 1.6),
                (1.0, 1.8),
              ]:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=full_sum_train_alias,
                  config_builder=config_builder,
                  checkpoint=full_sum_checkpoint,
                  checkpoint_aliases=("best-4-avg",),
                  att_readout_scale=att_readout_scale,
                  h_t_readout_scale=h_t_readout_scale,
                )

            if alias in ["v1_long_two-stage", "v1_long_two-stage-rand-seed-1234"]:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=full_sum_train_alias,
                config_builder=config_builder,
                checkpoint=full_sum_checkpoint,
                checkpoint_aliases=("last",),
                # only_do_analysis=True,
                calc_search_errors=True,
                run_analysis=True,
              )

              if alias == "v1_long_two-stage":
                params = [
                  # w/ LM + ILM
                  (0.8, 1.0, 0.1, 0.05),
                  (0.8, 1.0, 0.2, 0.05),
                  (0.8, 1.0, 0.3, 0.05),
                  (0.8, 1.0, 0.4, 0.05),
                  (0.8, 1.0, 0.5, 0.05),
                  (0.8, 1.0, 0.6, 0.05),
                  (0.8, 1.0, 0.7, 0.05),
                  #
                  (0.8, 1.0, 0.1, 0.1),
                  (0.8, 1.0, 0.2, 0.1),
                  (0.8, 1.0, 0.3, 0.1),
                  (0.8, 1.0, 0.4, 0.1),
                  (0.8, 1.0, 0.5, 0.1),
                  #
                  (0.8, 1.0, 0.5, 0.2),
                ]
                # dev-other: 5.5
                external_aed_checkpoint_path = "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/returnn/training/ReturnnTrainingJob.VNhMCmbnAUcd/output/models/epoch.2000.pt"
                external_aed_mini_lstm_checkpoint = "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/returnn/training/ReturnnTrainingJob.K1W7qayIjuM1/output/models/epoch.010.pt"
                external_aed_mini_lstm_checkpoint = PtCheckpoint(Path(external_aed_mini_lstm_checkpoint))
                ilm_type = "mini_att"
              else:
                params = [
                  (0.4, 1.0, 0.0, 0.0),
                  (0.5, 1.0, 0.0, 0.0),
                  (0.6, 1.0, 0.0, 0.0),
                  (0.7, 1.0, 0.0, 0.0),
                  (0.8, 1.0, 0.0, 0.0),
                  (0.9, 1.0, 0.0, 0.0),
                  (1.0, 1.0, 0.0, 0.0),
                  (1.1, 1.0, 0.0, 0.0),
                  #
                  (0.1, 0.9, 0.0, 0.0),
                  (0.2, 0.8, 0.0, 0.0),
                  (0.3, 0.7, 0.0, 0.0),
                  (0.4, 0.6, 0.0, 0.0),
                  (0.5, 0.5, 0.0, 0.0),
                  (0.6, 0.4, 0.0, 0.0),
                  (0.7, 0.3, 0.0, 0.0),
                  (0.8, 0.2, 0.0, 0.0),
                  (0.9, 0.1, 0.0, 0.0),
                  #
                  (0.8, 1.1, 0.0, 0.0),
                  (0.8, 1.2, 0.0, 0.0),
                  (0.8, 1.3, 0.0, 0.0),
                  #
                  (0.5, 1.1, 0.0, 0.0),
                  (0.5, 1.2, 0.0, 0.0),
                  (0.5, 1.3, 0.0, 0.0),
                ]
                # dev-other: 5.4
                external_aed_checkpoint_path = "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_core/returnn/training/ReturnnTrainingJob.czqZZvX66f4j/output/models/epoch.2000.pt"
                external_aed_mini_lstm_checkpoint = None
                ilm_type = "zero_att"

              for base_scale, aed_scale, lm_scale, ilm_scale in params:
                lm_alias = "1k_max-seq-length-112_24-layers_1024-dim"
                corpus_keys = ["dev-other"]
                if base_scale == 0.5 and aed_scale == 0.5 and alias == "v1_long_two-stage-rand-seed-1234" and lm_scale == 0.0:
                  corpus_keys.append("test-other")

                recog.center_window_returnn_frame_wise_beam_search(
                  alias=full_sum_train_alias,
                  config_builder=config_builder,
                  corpus_keys=corpus_keys,
                  checkpoint=full_sum_checkpoint,
                  checkpoint_aliases=("last",),
                  base_model_scale=base_scale,
                  external_aed_opts={
                    "checkpoint": PtCheckpoint(Path(external_aed_checkpoint_path)),
                    "scale": aed_scale,
                    "mini_lstm_checkpoint": external_aed_mini_lstm_checkpoint,
                  },
                  blank_scale=base_scale,
                  emit_scale=base_scale,
                  lm_type="trafo",
                  lm_scale_list=(lm_scale,),
                  ilm_scale_list=(ilm_scale,),
                  ilm_type=ilm_type,
                  lm_alias=lm_alias,
                  lm_checkpoint=lm_checkpoints[lm_alias],
                  add_lm_eos_to_non_blank_end_hyps=True,
                  lm_eos_scale=1.0,
                  subtract_ilm_eos_score=False,
                  sbatch_args=["-p", "gpu_48gb,gpu_24gb_preemptive,gpu_11gb"] if lm_scale == 0.0 else ["-p", "gpu_48gb,gpu_24gb_preemptive"],
                  batch_size=None if lm_scale == 0.0 else 20_000,
                  time_rqmt=None if lm_scale == 0.0 else 2,
                )

              if alias == "v1_long_two-stage-rand-seed-1234":
                for base_scale, external_transducer_scale, lm_scale, ilm_scale in [
                  (1.0, 0.3, 0.0, 0.0),
                  (1.0, 0.4, 0.0, 0.0),
                  (1.0, 0.5, 0.0, 0.0),
                  (1.0, 0.6, 0.0, 0.0),
                  (1.0, 0.7, 0.0, 0.0),
                  (1.0, 0.8, 0.0, 0.0),
                  (1.0, 0.9, 0.0, 0.0),
                  (0.8, 1.0, 0.0, 0.0),
                ]:
                  corpus_keys = ["dev-other"]
                  if base_scale == 0.8 and external_transducer_scale == 1.0:
                    corpus_keys.append("test-other")

                  lm_alias = "1k_max-seq-length-112_24-layers_1024-dim"
                  recog.center_window_returnn_frame_wise_beam_search(
                    alias=full_sum_train_alias,
                    config_builder=config_builder,
                    corpus_keys=corpus_keys,
                    checkpoint=full_sum_checkpoint,
                    checkpoint_aliases=("last",),
                    base_model_scale=base_scale,
                    external_transducer_opts={
                      # dev-other: 5.7
                      "checkpoint": PtCheckpoint(Path(
                        "/work/asr3/zeyer/schmitt/sisyphus_work_dirs/segmental_models_2022_23_rf/i6_core/returnn/training/ReturnnTrainingJob.6HvxwoXK7mFE/output/models/epoch.800.pt")),
                      "scale": external_transducer_scale,
                    },
                    blank_scale=base_scale,
                    emit_scale=base_scale,
                    lm_type="trafo",
                    lm_scale_list=(lm_scale,),
                    ilm_scale_list=(ilm_scale,),
                    ilm_type="zero_att",
                    lm_alias=lm_alias,
                    lm_checkpoint=lm_checkpoints[lm_alias],
                    add_lm_eos_to_non_blank_end_hyps=True,
                    lm_eos_scale=1.0,
                    subtract_ilm_eos_score=False,
                    sbatch_args=["-p", "gpu_48gb,gpu_24gb_preemptive"],
                    batch_size=None if lm_scale == 0.0 else 20_000,
                    time_rqmt=None if lm_scale == 0.0 else 2,
                  )

            if "long" in alias and alias not in ["v1_long_two-stage", "v2_long_two-stage"] and not use_sep_h_t_readout:
              if win_size is None and (
                use_current_frame_in_readout or
                use_current_frame_in_readout_w_double_gate or
                use_current_frame_in_readout_w_gate
              ):
                ilm_type = "mini_att"

                params = [
                  # (0.4, 0.1, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.4, 0.15, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.4, 0.2, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.4, 0.25, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.5, 0.1, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.5, 0.15, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.5, 0.2, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.5, 0.25, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.6, 0.1, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.6, 0.15, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.6, 0.2, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  # (0.6, 0.25, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.54, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                ]
              else:
                ilm_type = "mini_att"

                params = [
                  (0.54, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.6, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.6, 0.3, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.66, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.68, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.7, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.72, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.74, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.7, 0.3, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.8, 0.3, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                  (0.8, 0.4, True, 1.0, False, "1k_max-seq-length-112_24-layers_1024-dim"),
                ]

              if use_sep_h_t_readout:
                att_readout_scale = 1.0
                h_t_readout_scale = 0.5
                checkpoint_alias = "best-4-avg"
              else:
                att_readout_scale = None
                h_t_readout_scale = None
                checkpoint_alias = "last"
              for (
                      lm_scale,
                      ilm_scale,
                      add_lm_eos_to_non_blank_end_hyps,
                      lm_eos_scale,
                      subtract_ilm_eos_score,
                      lm_alias,
              ) in params:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=full_sum_train_alias,
                  config_builder=config_builder,
                  checkpoint=full_sum_checkpoint,
                  checkpoint_aliases=(checkpoint_alias,),
                  lm_type="trafo",
                  lm_scale_list=(lm_scale,),
                  ilm_scale_list=(ilm_scale,),
                  ilm_type=ilm_type,
                  lm_alias=lm_alias,
                  lm_checkpoint=lm_checkpoints[lm_alias],
                  add_lm_eos_to_non_blank_end_hyps=add_lm_eos_to_non_blank_end_hyps,
                  lm_eos_scale=lm_eos_scale,
                  subtract_ilm_eos_score=subtract_ilm_eos_score,
                  att_readout_scale=att_readout_scale,
                  h_t_readout_scale=h_t_readout_scale,
                  # batch_size=10_000 if win_size is None else None,
                  sbatch_args=None if lm_scale == 0.0 else ["-p", "gpu_48gb,gpu_24gb_preemptive"],
                  time_rqmt=None if lm_scale == 0.0 else 2,
                )

            if "long" in alias:
              if use_sep_h_t_readout:
                att_readout_scale = 1.0
                h_t_readout_scale = 0.5
                checkpoint_alias = "best-4-avg"
                best_scales = (None, None)
              elif alias in ["v1_long_two-stage", "v1_long_two-stage-rand-seed-1234"]:
                best_scales = (0.7, 0.4)
                checkpoint_alias = "last"
                att_readout_scale = None
                h_t_readout_scale = None
              else:
                att_readout_scale = None
                h_t_readout_scale = None
                checkpoint_alias = "last"
                best_scales = (None, None)

              # lm_scale, ilm_scale = best_scales

              for lm_scale, ilm_scale in [best_scales, (0.0, 0.0)]:
                if (lm_scale, ilm_scale) != (None, None):
                  for beam_size in [12, 84] if lm_scale > 0.0 else [12]:
                    for ilm_type in ("mini_att", "zero_att"):
                      recog.center_window_returnn_frame_wise_beam_search(
                        alias=full_sum_train_alias,
                        config_builder=config_builder,
                        checkpoint=full_sum_checkpoint,
                        checkpoint_aliases=(checkpoint_alias,),
                        lm_type="trafo",
                        lm_scale_list=(lm_scale,),
                        ilm_scale_list=(ilm_scale,),
                        ilm_type=ilm_type,
                        lm_alias="1k_max-seq-length-112_24-layers_1024-dim",
                        lm_checkpoint=lm_checkpoints["1k_max-seq-length-112_24-layers_1024-dim"],
                        add_lm_eos_to_non_blank_end_hyps=True,
                        lm_eos_scale=1.0,
                        subtract_ilm_eos_score=False,
                        att_readout_scale=att_readout_scale,
                        h_t_readout_scale=h_t_readout_scale,
                        batch_size=15_000 if beam_size == 12 else 6_000,
                        sbatch_args=None if lm_scale == 0.0 else ["-p", "gpu_48gb,gpu_24gb_preemptive"],
                        time_rqmt=None if lm_scale == 0.0 else 2,
                        corpus_keys=("test-other",),
                        beam_size_list=(beam_size,),
                      )

            for epoch, chckpt in full_sum_checkpoint["checkpoints"].items():
              if epoch in keep_epochs_full_sum and not use_sep_h_t_readout:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=full_sum_train_alias,
                  config_builder=config_builder,
                  checkpoint=chckpt,
                  checkpoint_aliases=(f"epoch-{epoch}",),
                )

  plot_gradient_wrt_enc11(
    data.analyze_gradients_jobs["baseline_v3_two-stage"]["fixed-path"]["v2"]["24gb-gpu"],
    alias=f"{base_alias}/gradients_lstm-att_wo-att-ctx-in-state_fixed-path_epoch-600"
  )
