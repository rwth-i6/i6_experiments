from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import data
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import lm_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import plot_gradient_wrt_enc11
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_CTC_ALIGNMENT

from i6_core.returnn.training import PtCheckpoint

from sisyphus import Path

def run_exps():
  # ------------------------------ Transducer ------------------------------

  data.analyze_gradients_jobs["baseline_v5_two-stage"] = {"fixed-path": {}, "full-sum": {}}
  n_full_epochs_fixed_path = 30
  n_full_epochs_full_sum = 15
  for (
    alias,
    win_size,
    use_trafo_att,
    use_trafo_att_wo_cross_att,
    accum_grad_multiple_step_,
    use_att_ctx_in_state,
    att_h_t_dropout,
    weight_decay,
    blank_decoder_version,
    blank_decoder_opts,
    n_full_epochs_fixed_path,
    n_full_epochs_full_sum,
    lr_scale,
    random_seed,
    use_current_frame_in_readout,
    use_weight_feedback,
  ) in [
    ("v1", 1, False, False, None, False, 0.0, 1e-6, 4, None, 30, 15, 1.0, None, False, False),  # standard transducer
    ("v1_long_two-stage", 1, False, False, None, False, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, False, False),  # standard transducer, longer training
    ("v1_long_two-stage_rand-seed-1234", 1, False, False, None, False, 0.0, 1e-6, 4, None, 60, 40, 1.0, 1234, False, False),  # standard transducer, longer training
    ("v1_accum1", 1, False, False, 1, False, 0.0, 1e-6, 4, None, 30, 15, 1.0, None, False, False),  # standard transducer, accum 1
    ("v1_accum1_reg_v3", 1, False, False, 1, False, 0.0, 0.01, 4, None, 30, 15, 1.0, None, False, False),  # standard transducer, accum 1, v3 regularization
    ("v1_accum1_reg_v3_lr-scale-2.0", 1, False, False, 1, False, 0.0, 0.01, 4, None, 30, 15, 2.0, None, False, False),  # standard transducer, accum 1, v3 regularization, lr scale 2.0
    ("v1_accum1_reg_v4_lr-scale-2.0", 1, False, False, 1, False, 0.0, 2e-2, 4, None, 30, 15, 2.0, None, False, False),  # standard transducer, accum 1, v3 regularization, lr scale 2.0
    ("v1_accum1_reg_v4", 1, False, False, 1, False, 0.0, 2e-2, 4, None, 30, 15, 1.0, None, False, False),  # standard transducer, accum 1, v4 regularization
    ("v1_accum1_reg_v5", 1, False, False, 1, False, 0.0, 4e-2, 4, None, 30, 15, 1.0, None, False, False),  # standard transducer, accum 1, v5 regularization
    ("v1_accum1_reg_v3_blank-drop-0.3", 1, False, False, 1, False, 0.0, 0.01, 4, {"dropout": 0.3}, 30, 15, 1.0, None, False, False),  # standard transducer, accum 1, v3 regularization, blank dropout
    ("v1_accum1_reg_v3_blank-v11", 1, False, False, 1, False, 0.0, 0.01, 11, None, 30, 15, 1.0, None, False, False),  # standard transducer, accum 1, v3 regularization, blank v11
    ("v2", None, True, False, None, False, 0.0, 1e-6, 4, None, 30, 15, 1.0, None, True, False),  # transducer with transformer attention
    ("v2_long_two-stage", None, True, False, None, False, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, True, False),  # transducer with transformer attention
    ("v3", None, True, True, None, False, 0.0, 1e-6, 4, None, 30, 15, 1.0, None, True, False),  # transducer with transformer w/o cross attention
    ("v3_long_two-stage", None, True, True, None, False, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, True, False),  # transducer with transformer w/o cross attention
    ("v4", None, False, False, None, True, 0.0, 1e-6, 4, None, 30, 15, 1.0, None, True, False),  # standard transducer with global LSTM att and att ctx in state
    ("v4_long_two-stage", None, False, False, None, True, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, True, False),  # standard transducer with global LSTM att and att ctx in state
    ("v5", None, True, False, None, False, 0.3, 1e-6, 4, None, 30, 15, 1.0, None, True, False),  # transducer with transformer attention with random gate
    ("v5_drop0.5", None, True, False, None, False, 0.5, 1e-6, 4, None, 30, 15, 1.0, None, True, False),  # transducer with transformer attention with random gate
    ("v6_long_first-stage", None, False, False, None, True, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, False, False),  # standard transducer with global LSTM att and att ctx in state and h_t not in readout
    ("v7_long_first-stage", None, False, False, None, True, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, False, True),  # standard transducer with global LSTM att and att ctx in state and wf and h_t not in readout
    ("v8_long_first-stage", None, False, False, None, False, 0.0, 1e-6, 4, None, 60, 40, 1.0, None, False, False),  # standard transducer with global LSTM att and h_t not in readout
  ]:
    gpu_mem_rqmts = [24]
    if alias == "v1":
      gpu_mem_rqmts.append(11)
    if alias in [
      "v1_accum1",
      "v4",
      "v1_accum1_reg_v3",
      "v1_accum1_reg_v4",
      "v1_accum1_reg_v5",
      "v1_accum1_reg_v3_blank-drop-0.3",
      "v1_accum1_reg_v3_blank-v11",
      "v1_accum1_reg_v3_lr-scale-2.0",
      "v1_accum1_reg_v4_lr-scale-2.0"
    ]:
      gpu_mem_rqmts = [11]

    if "long" in alias:
      gpu_mem_rqmts = [24]

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
        batch_size = 15_000 if alias != "v4" else 12_000
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20 // 4
        n_epochs_full_sum = n_full_epochs_full_sum * 20 // 4

      if accum_grad_multiple_step_ is not None:
        accum_grad_multiple_step = accum_grad_multiple_step_

      for model_alias, config_builder in get_config_builder.center_window_att_baseline_rf(
              win_size_list=(win_size,),
              blank_decoder_version=blank_decoder_version,
              use_att_ctx_in_state=use_att_ctx_in_state,
              use_weight_feedback=use_weight_feedback,
              bpe_vocab_size=1056,
              use_correct_dim_tags=True,
              behavior_version=21,
              use_trafo_att=use_trafo_att,
              use_current_frame_in_readout=use_current_frame_in_readout,
              use_trafo_att_wo_cross_att=use_trafo_att_wo_cross_att,
              blank_decoder_opts=blank_decoder_opts,
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
                  "init_lr": 1e-5 * lr_scale,
                  "peak_lr": 1e-3 * lr_scale,
                  "lr2": 1e-5 * lr_scale,
                },
                att_h_t_dropout=att_h_t_dropout,
                weight_decay=weight_decay,
                random_seed=random_seed,
        ):
          checkpoint_aliases = ("last",)
          if alias in (
                  "v1_accum1_reg_v4",
                  "v1_accum1_reg_v5",
                  "v1_accum1_reg_v3_lr-scale-2.0",
                  "v1_accum1_reg_v4_lr-scale-2.0"
          ):
            checkpoint_aliases = ("last", "best", "best-4-avg")
          if "long" in alias:
            checkpoint_aliases = ("last", "best", "best-4-avg")

          recog.center_window_returnn_frame_wise_beam_search(
            alias=fixed_path_train_alias,
            config_builder=config_builder,
            checkpoint=fixed_path_checkpoint,
            checkpoint_aliases=checkpoint_aliases,
          )
          if win_size is None and "long" in alias:
            pipeline = recog.center_window_returnn_frame_wise_beam_search(
              alias=fixed_path_train_alias,
              config_builder=config_builder,
              checkpoint=fixed_path_checkpoint,
              checkpoint_aliases=("last", "best-4-avg"),
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
            data.analyze_gradients_jobs["baseline_v5_two-stage"]["fixed-path"][alias] = pipeline.decoding_exps[0].analyze_gradients_job

          if alias == "v1" and gpu_mem_rqmt == 24:
            for lm_scale, ilm_scale in [
              (0.54, 0.4),
              (0.6, 0.4),
              (0.7, 0.4),
              (0.7, 0.3),
            ]:
              lm_alias = "1k_max-seq-length-75_24-layers_512-dim"
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

          for epoch, chckpt in fixed_path_checkpoint["checkpoints"].items():
            if epoch in keep_epochs_fixed_path:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=fixed_path_train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
              )
              if (
                      win_size is None and not use_trafo_att and epoch in [45, 60, 150, 180, 240, 600]
              ) or (
                use_trafo_att and epoch in [60, 180, 600, 150]
              ):
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

        keep_epochs_step_full_sum = n_epochs_full_sum // 10
        keep_epochs_full_sum = list(range(keep_epochs_step_full_sum, n_epochs_full_sum, keep_epochs_step_full_sum))
        params = []
        if gpu_mem_rqmt == 24 and alias in ("v1", "v2", "v3", "v4"):
          params += [(3e-4, False)]
          if alias == "v1":
            params += [(3e-4, True)]

        if "long" in alias and alias not in ["v6_long_first-stage", "v7_long_first-stage", "v8_long_first-stage"]:
          params += [(1e-4, True)]
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
                  weight_decay=weight_decay,
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
            recog.center_window_returnn_frame_wise_beam_search(
              alias=full_sum_train_alias,
              config_builder=config_builder,
              checkpoint=full_sum_checkpoint,
            )

            if alias == "v1_long_two-stage" and gpu_mem_rqmt == 24:
              recog.center_window_returnn_frame_wise_beam_search(
                alias=full_sum_train_alias,
                config_builder=config_builder,
                checkpoint=full_sum_checkpoint,
                checkpoint_aliases=("last",),
                # only_do_analysis=True,
                calc_search_errors=True,
                run_analysis=True,
              )

              for base_scale, aed_scale, lm_scale, ilm_scale in [
                (0.1, 0.9, 0.0, 0.0),
                (0.2, 0.8, 0.0, 0.0),
                (0.3, 0.7, 0.0, 0.0),
                (0.4, 0.6, 0.0, 0.0),
                (0.5, 0.5, 0.0, 0.0),
                (0.6, 0.4, 0.0, 0.0),
                (0.7, 0.3, 0.0, 0.0),
                (0.8, 0.2, 0.0, 0.0),
                (0.9, 0.1, 0.0, 0.0),
                (0.2, 1.0, 0.0, 0.0),
                (0.4, 1.0, 0.0, 0.0),
                (0.5, 1.0, 0.0, 0.0),
                (0.6, 1.0, 0.0, 0.0),
                (0.7, 1.0, 0.0, 0.0),
                (0.8, 1.0, 0.0, 0.0),
                (0.9, 1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0, 0.0),
                (1.1, 1.0, 0.0, 0.0),
              ]:
                lm_alias = "1k_max-seq-length-112_24-layers_1024-dim"
                corpus_keys = ["dev-other"]
                # if base_scale == 0.5 and aed_scale == 1.0:
                #   corpus_keys.append("test-other")
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=full_sum_train_alias,
                  config_builder=config_builder,
                  corpus_keys=corpus_keys,
                  checkpoint=full_sum_checkpoint,
                  checkpoint_aliases=("best-4-avg",),
                  base_model_scale=base_scale,
                  external_aed_opts={
                    "checkpoint": PtCheckpoint(Path(
                      "/u/schmitt/experiments/segmental_models_2022_23_rf/work/i6_core/returnn/training/ReturnnTrainingJob.VNhMCmbnAUcd/output/models/epoch.2000.pt")),
                    "scale": aed_scale,
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
                )

              for lm_scale, ilm_scale in [
                (0.54, 0.4),
                (0.6, 0.4),
                (0.6, 0.3),
                (0.7, 0.4),
                (0.7, 0.3),
                (0.8, 0.3),
                (0.8, 0.4),
                (0.9, 0.3),
                (0.9, 0.4),
                (0.9, 0.5),
                (0.9, 0.6),
              ]:
                for lm_alias in [
                  "1k_max-seq-length-112_24-layers_1024-dim",
                ]:
                  recog.center_window_returnn_frame_wise_beam_search(
                    alias=full_sum_train_alias,
                    config_builder=config_builder,
                    checkpoint=full_sum_checkpoint,
                    checkpoint_aliases=("best-4-avg",),
                    lm_type="trafo",
                    lm_scale_list=(lm_scale,),
                    ilm_scale_list=(ilm_scale,),
                    ilm_type="mini_att",
                    lm_alias=lm_alias,
                    lm_checkpoint=lm_checkpoints[lm_alias],
                  )


            if "long" in alias:
              if alias in ["v1_long_two-stage",]:
                best_scales = (0.7, 0.3)
                checkpoint_alias = "best-4-avg"
              else:
                checkpoint_alias = "last"
                best_scales = (None, None)

              # lm_scale, ilm_scale = best_scales

              if best_scales != (None, None):
                for lm_scale, ilm_scale in [best_scales, (0.0, 0.0)]:
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
                        batch_size=15_000 if beam_size == 12 else 6_000,
                        sbatch_args=None if lm_scale == 0.0 else ["-p", "gpu_48gb,gpu_24gb_preemptive"],
                        time_rqmt=None if lm_scale == 0.0 else 2,
                        corpus_keys=("test-other",),
                        beam_size_list=(beam_size,),
                      )

            for epoch, chckpt in full_sum_checkpoint["checkpoints"].items():
              if epoch in keep_epochs_full_sum:
                recog.center_window_returnn_frame_wise_beam_search(
                  alias=full_sum_train_alias,
                  config_builder=config_builder,
                  checkpoint=chckpt,
                  checkpoint_aliases=(f"epoch-{epoch}",),
                )
