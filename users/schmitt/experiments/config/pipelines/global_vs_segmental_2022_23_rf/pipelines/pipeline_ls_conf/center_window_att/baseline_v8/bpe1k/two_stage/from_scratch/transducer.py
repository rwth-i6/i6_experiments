from i6_core.returnn import PtCheckpoint
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v8 import (
  get_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v8.alias import alias as base_alias
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

  data.analyze_gradients_jobs["baseline_v8_two-stage"] = {"fixed-path": {}, "full-sum": {}}

  n_full_epochs_fixed_path = 100
  for (
    alias,
    win_size,
    win_step_size,
    batch_size,
    use_weight_feedback,
    use_att_ctx_in_state,
  ) in [
    ("v1", 1, 1, 30_000, False, True),  # standard RNN-T transducer
    ("v2", 25, 25, 30_000, True, True),  # chunked AED
  ]:
    data.analyze_gradients_jobs["baseline_v8_two-stage"]["fixed-path"][alias] = {}

    gpu_mem_rqmts = [24]

    for gpu_mem_rqmt in gpu_mem_rqmts:
      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 1
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20
      else:
        use_mgpu = True
        accum_grad_multiple_step = 1
        n_epochs_fixed_path = n_full_epochs_fixed_path * 20 // 4

      model_alias, config_builder = get_config_builder.center_window_att_baseline_rf(
        win_size=win_size,
        use_att_ctx_in_state=use_att_ctx_in_state,
        use_weight_feedback=use_weight_feedback,
        bpe_vocab_size=1056,
        window_step_size=win_step_size,
      )
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
        if alias == "v1":
          checkpoint_aliases = ("best",)
        else:
          checkpoint_aliases = ("last",)
        for length_normalization_exponent in [0.0, 1.0]:
          recog.center_window_returnn_frame_wise_beam_search(
            alias=fixed_path_train_alias,
            config_builder=config_builder,
            checkpoint=fixed_path_checkpoint,
            checkpoint_aliases=checkpoint_aliases,
            use_recombination=None,
            length_normalization_exponent=length_normalization_exponent,
          )

        recog.center_window_returnn_frame_wise_beam_search(
          alias=fixed_path_train_alias,
          config_builder=config_builder,
          checkpoint=fixed_path_checkpoint,
          use_recombination=None,
          corpus_keys=("test-other",),
          checkpoint_aliases=("last",) if alias == "v2" else ("best",)
        )

        if alias in ["v1", "v2"]:
          if alias == "v1":
            params = [
              (0.0, 0.0),
              (0.1, 0.0),
              (0.2, 0.0),
              (0.3, 0.0),
              (0.4, 0.0),
              (0.4, 0.1),
              (0.4, 0.2),
              (0.4, 0.3),
              (0.4, 0.4),
              (0.5, 0.0),
              (0.5, 0.1),
              (0.5, 0.2),
              (0.5, 0.3),
              (0.5, 0.4),
            ]
          else:
            params = [
              (0.0, 0.0),
              (0.1, 0.0),
              (0.2, 0.0),
              (0.3, 0.0),
              (0.4, 0.0),
              (0.4, 0.1),
              (0.4, 0.2),
              (0.4, 0.3),
              # (0.4, 0.4),
              # (0.5, 0.0),
              # (0.5, 0.1),
              # (0.5, 0.2),
              # (0.5, 0.3),
              # (0.5, 0.4),
            ]
          for lm_scale, ilm_scale in params:
            length_normalization_exponents = [1.0]
            for length_normalization_exponent in length_normalization_exponents:
              corpus_keys = ["dev-other"]
              if lm_scale == 0.4 and ilm_scale == 0.0:
                corpus_keys += ["test-other"]
              lm_alias = "1k_max-seq-length-112_24-layers_1024-dim"
              recog.center_window_returnn_frame_wise_beam_search(
                alias=fixed_path_train_alias,
                config_builder=config_builder,
                checkpoint=fixed_path_checkpoint,
                checkpoint_aliases=checkpoint_aliases,
                lm_type="trafo",
                lm_scale_list=(lm_scale,),
                ilm_scale_list=(ilm_scale,),
                ilm_type="zero_att",
                lm_alias=lm_alias,
                lm_checkpoint=lm_checkpoints[lm_alias],
                use_recombination=None,
                sbatch_args=["-p", "gpu_48gb,gpu_24gb,gpu_11gb"],
                length_normalization_exponent=length_normalization_exponent,
                batch_size=None if alias == "v2" else 10_000,
                subtract_ilm_eos_score=True,
                corpus_keys=corpus_keys,
              )

        for epoch, chckpt in fixed_path_checkpoint["checkpoints"].items():
          if epoch in keep_epochs_fixed_path:
            recog.center_window_returnn_frame_wise_beam_search(
              alias=fixed_path_train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              use_recombination=None,
            )
            # recog.center_window_returnn_frame_wise_beam_search(
            #   alias=fixed_path_train_alias,
            #   config_builder=config_builder,
            #   checkpoint=chckpt,
            #   checkpoint_aliases=(f"epoch-{epoch}",),
            #   run_analysis=True,
            #   only_do_analysis=True,
            #   analyze_gradients=True,
            #   analsis_analyze_gradients_plot_log_gradients=True,
            #   analysis_analyze_gradients_plot_encoder_layers=True,
            #   analysis_ground_truth_hdf=LibrispeechBPE1056_CTC_ALIGNMENT.alignment_paths["train"],
            #   att_weight_seq_tags=[
            #     "train-other-960/1246-124548-0042/1246-124548-0042",
            #     "train-other-960/40-222-0033/40-222-0033",
            #     "train-other-960/103-1240-0038/103-1240-0038",
            #   ],
            #   corpus_keys=("train",),
            # )
