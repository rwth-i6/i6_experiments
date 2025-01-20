from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
import os
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_WORD_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import LibrispeechBPE1056_LABELS
from i6_experiments.users.schmitt.visualization.visualization import PlotAttentionWeightsJobV2, PlotSelfAttentionWeightsOverEpochsJob
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import lm_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import plot_gradient_wrt_enc11, plot_diff_models

from sisyphus import Path, tk
from sisyphus.delayed_ops import DelayedFormat


def run_exps():
  flipped_att_weights_evolution = []
  flipped_att_weights_evolution_epochs = [374, 484, 490, 500]
  analyze_gradients_jobs = {}
  for (
          alias,
          random_seed,
          disable_self_att_until_epoch,
          ctc_aux_loss_layers,
          conformer_wo_final_layer_norm_per_layer,
          conformer_num_layers,
          conformer_out_dim,
          conformer_wo_convolution,
          conformer_wo_rel_pos_enc,
          conformer_w_abs_pos_enc,
          keep_epochs,
          gpu_mem_rqmt,
          enc_ctx_layer,
          hard_att_opts,
          conformer_conv_w_zero_padding,
          use_feed_forward_encoder,
          conv_frontend_w_zero_padding,
          cutoff_initial_silence,
          use_speed_pert_w_flip,
          weight_decay,
          accum_grad_multiple_step_,
  ) in [
    # ["v3_big", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24], # v3_big: same as v2, but on 24gb GPU with batch size 40k
    ["v3_wo-wf-w-ctx-in-state", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],
    ["v3_wo-wf-wo-ctx-in-state", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],
    ["v3_w-wf-w-ctx-in-state", 9999, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],
    ["v3_wo-wf-w-ctx-in-state_ctc", None, None, (4, 8), False, 12, 512, False, False, False, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],
    ["v3_wo-wf-wo-ctx-in-state_ctc", None, None, (4, 8), False, 12, 512, False, False, False, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],
    ["v3_w-wf-w-ctx-in-state_ctc", 9999, None, (4, 8), False, 12, 512, False, False, False, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],
    ["v3_rand-9999", 9999, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand - flipped
    ["v3_rand-1234", 1234, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand -
    ["v3_rand-1111", 1111, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_rand-4321", 4321, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_rand-5678", 5678, None, None, False, 12, 512, False, False, False, list(range(10, 80, 10)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_big_rand-5678", 5678, None, None, False, 12, 512, False, False, False, list(range(20, 200, 20)), 24, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_rand-8765", 8765, None, None, False, 12, 512, False, False, False, list(range(10, 80, 10)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_rand-2222", 2222, None, None, False, 12, 512, False, False, False, list(range(10, 80, 10)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_big_rand-2222", 2222, None, None, False, 12, 512, False, False, False, list(range(20, 200, 20)), 24, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v3_rand-3333", 3333, None, None, False, 12, 512, False, False, False, list(range(10, 80, 10)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v3_big_rand
    ["v5", None, 21, None, False, 12, 512, False, False, False, list(range(61)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v5_big: same as v3_big, but enable self attention only after 20 sub-epochs (1 full epoch)
    ["v5_rand-1234", 1234, 21, None, False, 12, 512, False, False, False, list(range(61)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v5_big: same as v3_big, but enable self attention only after 20 sub-epochs (1 full epoch)
    ["v5_big_rand-1234", 1234, 21, None, False, 12, 512, False, False, False, list(range(20, 200, 20)), 24, None, None, False, False, False, False, False, 1e-6, None],  # v5_big: same as v3_big, but enable self attention only after 20 sub-epochs (1 full epoch)
    ["v6_big", None, None, None, False, 12, 512, False, False, True, list(range(1, 240)), 24, None, None, False, False, False, False, False, 1e-6, None],  # v6_big: same as v3_big, but use both absolute and relative positional encodings
    ["v6", None, None, None, False, 12, 512, False, False, True, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None], # v6_big: same as v3_big, but use both absolute and relative positional encodings
    # ["v7_big", None, None, None, True, 12, 512, False, False, False, [121, 131, 141], 24, None, None, False],  # v7_big: same as v3_big, but do not use final layer norm in conformer encoder layers
    ["v7", None, None, None, True, 12, 512, False, False, False, [121, 131, 141], 11, None, None, False, False, False, False, False, 1e-6, None],  # v7: same as v3_big, but do not use final layer norm in conformer encoder layers
    ["v8", None, None, (4, 8), False, 12, 512, False, False, False, list(range(51)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v8_big: same as v3_big, but use CTC aux loss
    ["v8_big", None, None, (4, 8), False, 12, 512, False, False, False, list(range(20, 200, 20)), 24, None, None, False, False, False, False, False, 1e-6, None],  # v8_big: same as v3_big, but use CTC aux loss
    ["v85_big", None, 21, (4, 8), False, 12, 512, False, False, False, list(range(20, 200, 20)), 24, None, None, False, False, False, False, False, 1e-6, None],  # v8_big: same as v3_big, but use CTC aux loss
    # ["v9_big", None, None, None, False, 17, 400, False, False, False, list(range(1, 240)), 24, None, None, False],  # v9_big: same as v3_big, but use 17 instead of 12 encoder layers and 400 instead of 512 output dim
    ["v9", None, None, None, False, 17, 400, False, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None], # v9: same as v3_big, but use 17 instead of 12 encoder layers and 400 instead of 512 output dim
    ["v9_rand-1234", 1234, None, None, False, 17, 400, False, False, False, list(range(1, 122)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v9: same as v3_big, but use 17 instead of 12 encoder layers and 400 instead of 512 output dim
    # ["v10", None, None, None, False, 12, 512, True, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, False], # v10_big: same as v3_big, but without convolution module in conformer encoder layers
    # ["v11_big", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24, "encoder_input", None, False], # v11_big: same as v3_big, but use encoder input as att keys
    ["v11", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, "encoder_input", None, False, False, False, False, False, 1e-6, None],  # v11_big: same as v3_big, but use encoder input as att keys
    # ["v12_big", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24, None, {"frame": "middle", "until_epoch": 100, "num_interpolation_epochs": 20}, False], # v12_big: same as v3_big, but use hard att on center frame until sub-epoch 100
    ["v12", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, {"frame": "middle", "until_epoch": 100, "num_interpolation_epochs": 20}, False, False, False, False, False, 1e-6, None],  # v12_big: same as v3_big, but use hard att on center frame until sub-epoch 100
    ["v13", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, None, True, False, False, False, False, 1e-6, None], # v13: same as v3_big, but set padding to zero before depthwise conv in conformer encoder layers
    # ["v14", None, None, None, False, 6, 512, False, False, False, list(range(1, 240)), 11, None, None, False, True, False, False, False],  # v14: same as v3_big, but use FF encoder with 6 layers -> not converged
    ["v15", None, None, None, False, 12, 512, False, True, False, list(range(1, 240)), 11, None, None, False, False, False, False, False, 1e-6, None],  # v15: same as v3, but without pos encoding
    ["v16", None, None, None, False, 12, 512, False, False, False, list(range(1, 120)), 11, None, None, True, False, True, False, False, 1e-6, None],  # v16: same as v3, but set padding to zero before depthwise conv in conformer encoder layers and before conv in frontend
    ["v17", None, None, None, False, 12, 512, False, False, False, list(range(1, 120)), 11, None, None, False, False, False, True, False, 1e-6, None],  # v17: same as v3, but cut off initial silence
    ["v19", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 11, None, None, False, False, False, False, True, 1e-6, None],  # v19: same as v3 but reverse audio
    ["v20", None, None, None, False, 12, 512, False, False, False, list(range(10, 100, 10)), 11, None, None, False, False, False, False, False, 0.01, 1],  # v20: same as v3 but with weight decay 1e-2 and grad accum 1
  ]:
    if "v3_wo-wf-w-ctx-in-state" in alias:
      use_weight_feedback = False
      use_att_ctx_in_state = True
    elif "v3_wo-wf-wo-ctx-in-state" in alias:
      use_weight_feedback = False
      use_att_ctx_in_state = False
    else:
      use_weight_feedback = True
      use_att_ctx_in_state = True
    for model_alias, config_builder in baseline.global_att_baseline_rf(
            use_weight_feedback=use_weight_feedback,
            use_att_ctx_in_state=use_att_ctx_in_state,
            label_type="bpe1056",
            conformer_wo_final_layer_norm_per_layer=conformer_wo_final_layer_norm_per_layer,
            conformer_num_layers=conformer_num_layers,
            conformer_out_dim=conformer_out_dim,
            conformer_wo_convolution=conformer_wo_convolution,
            conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
            conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
            enc_ctx_layer=enc_ctx_layer,
            conformer_conv_w_zero_padding=conformer_conv_w_zero_padding,
            use_feed_forward_encoder=use_feed_forward_encoder,
            hard_att_opts=hard_att_opts,
            conv_frontend_w_zero_padding=conv_frontend_w_zero_padding,
    ):
      if alias == "v8":
        use_mgpu = False
        accum_grad_multiple_step = 4
        batch_size = 15_000
        n_epochs = 500
      else:
        if gpu_mem_rqmt == 24:
          use_mgpu = False
          accum_grad_multiple_step = 2
          batch_size = 30_000 if alias == "v9_big" else 35_000
          n_epochs = 2_000
        else:
          use_mgpu = True
          accum_grad_multiple_step = 4
          batch_size = 12_000 if alias == "v9" else 15_000
          n_epochs = 500

      if any([str_ in alias for str_ in ["v3_wo-wf-w-ctx-in-state", "v3_wo-wf-wo-ctx-in-state", "v3_w-wf-w-ctx-in-state"]]):
        n_epochs = 1200

      if accum_grad_multiple_step_ is not None:
        accum_grad_multiple_step = accum_grad_multiple_step_

      for train_alias, checkpoint in train.train_global_att(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs=n_epochs,
              batch_size=batch_size,
              keep_epochs=keep_epochs,
              gpu_mem_rqmt=gpu_mem_rqmt,
              accum_grad_multiple_step=accum_grad_multiple_step,
              use_mgpu=use_mgpu,
              use_torch_amp=False,
              filter_data_len=19.5 * 16_000,
              random_seed=random_seed,
              disable_enc_self_att_until_epoch=disable_self_att_until_epoch,
              ctc_aux_loss_layers=ctc_aux_loss_layers,
              hard_att_opts=hard_att_opts,
              cutoff_initial_silence=cutoff_initial_silence,
              use_speed_pert_w_flip=use_speed_pert_w_flip,
              weight_decay=weight_decay,
      ):
        corpus_keys = ["dev-other"]
        checkpoint_aliases = ("last", "best", "best-4-avg")
        if "big" in alias and alias != "v8_big":
          corpus_keys += ["dev-clean", "test-clean", "test-other"]
          if alias in ["v3_big_rand-5678", "v5_big_rand-1234", "v85_big"]:
            checkpoint_aliases = ("last",)
          elif alias in ["v3_big_rand-2222",]:
            checkpoint_aliases = ("best",)
          elif alias in ["v6_big"]:
            checkpoint_aliases = ("best-4-avg",)
        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          corpus_keys=corpus_keys,
          checkpoint_aliases=checkpoint_aliases,
        )
        pipeline = recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          checkpoint_aliases=("last", "best"),
          run_analysis=True,
          only_do_analysis=True,
          analyze_gradients=True,
          att_weight_seq_tags=[
            "train-other-960/1246-124548-0042/1246-124548-0042",
            "train-other-960/40-222-0033/40-222-0033",
            "train-other-960/103-1240-0038/103-1240-0038",
          ],
          corpus_keys=("train",),
          analsis_analyze_gradients_plot_log_gradients=alias in [
            "v3_wo-wf-wo-ctx-in-state_ctc", "v3_wo-wf-w-ctx-in-state_ctc", "v3_w-wf-w-ctx-in-state_ctc"
          ]
        )
        analyze_gradients_jobs[alias] = pipeline.decoding_exps[0].analyze_gradients_job

        if alias in ["v20", "v5_big_rand-1234", "v85_big", "v3_rand-2222", "v5"]:
          for lm_scale, ilm_scale in [
            (0.54, 0.4),
            # (0.5, 0.4),
          ]:
            lm_alias = "1k_max-seq-length-112_24-layers_1024-dim"
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=checkpoint,
              corpus_keys=("dev-other",),
              checkpoint_aliases=("last",),
              lm_type="trafo",
              lm_scale_list=(lm_scale,),
              ilm_scale_list=(ilm_scale,),
              ilm_type="mini_att",
              lm_alias=lm_alias,
              lm_checkpoint=lm_checkpoints[lm_alias],
              behavior_version=21,  # otherwise trafo lm logits weight dims are flipped apparently
            )

        if alias in ["v3_big_rand-5678", "v3_big_rand-2222"]:
          analysis_epochs = [100, 160, 180]
        else:
          analysis_epochs = []
        for epoch, chckpt in checkpoint["checkpoints"].items():
          if epoch in analysis_epochs:
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              corpus_keys=("train",),
              run_analysis=True,
              analyze_gradients=True,
              only_do_analysis=True,
              att_weight_seq_tags=["train-other-960/40-222-0033/40-222-0033"],
              analysis_ref_alignment_opts={
                "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
                "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
                "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
              },
            )
        #   if epoch in [61, 225] and alias == "v16":
        #     if epoch == 61:
        #       input_layer_names = ["encoder_input", "frontend_input"]
        #     else:
        #       input_layer_names = ["frontend_input"]
        #     for input_layer_name in input_layer_names:
        #       recog.global_att_returnn_label_sync_beam_search(
        #         alias=train_alias,
        #         config_builder=config_builder,
        #         checkpoint=chckpt,
        #         checkpoint_aliases=(f"epoch-{epoch}",),
        #         run_analysis=True,
        #         analysis_dump_gradients=True,
        #         only_do_analysis=True,
        #         corpus_keys=("train",),
        #         att_weight_seq_tags=None,
        #         analysis_dump_gradients_input_layer_name=input_layer_name,
        #       )
        #
        #   if (alias == "v3_rand-4321" and epoch in [
        #     10,
        #     20, 30, 40, 50, 60
        #   ]) or (alias == "v8" and epoch in [11, 21, 31, 41, 51, 61]) or (
        #     alias == "v3_rand-9999" and epoch in [10, 20, 30, 40, 50, 60, 70, 80]
        #   ) or (alias == "v6" and epoch in list(range(51)) + [60]):
        #     seq_tags = []
        #     if alias == "v3_rand-9999" or (alias == "v8" and epoch == 61):
        #       seq_tags.append("train-other-960/1246-124548-0042/1246-124548-0042")
        #     else:
        #       seq_tags.append("train-other-960/1578-6379-0013/1578-6379-0013")
        #
        #     recog.global_att_returnn_label_sync_beam_search(
        #       alias=train_alias,
        #       config_builder=config_builder,
        #       checkpoint=chckpt,
        #       checkpoint_aliases=(f"epoch-{epoch}",),
        #       corpus_keys=("train",),
        #       run_analysis=True,
        #       analyze_gradients=True,
        #       only_do_analysis=True,
        #       att_weight_seq_tags=seq_tags,
        #       analysis_ref_alignment_opts={
        #         "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
        #         "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
        #         "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
        #       },
        #       analysis_analyze_gradients_plot_encoder_layers=True,
        #       analsis_analyze_gradients_plot_log_gradients=True,
        #     )
        #
          if (alias == "v6" and epoch in flipped_att_weights_evolution_epochs):
            seq_tags = ["train-other-960/40-222-0033/40-222-0033"]

            pipeline = recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              corpus_keys=("train",),
              run_analysis=True,
              analyze_gradients=True,
              only_do_analysis=True,
              att_weight_seq_tags=seq_tags,
              analysis_ref_alignment_opts={
                "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
                "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
                "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
              },
            )
            flipped_att_weights_evolution.append(pipeline.decoding_exps[0].analyze_gradients_job)

          if (alias == "v6" and epoch in [10, 20, 30, 40, 50, 60] + list(range(20, 51))):
            seq_tags_list = [["train-other-960/40-222-0033/40-222-0033"]]
            if epoch in [40, 45]:
              seq_tags_list += [
                [
                  "train-other-960/103-1240-0038/103-1240-0038",
                  "train-other-960/103-1240-0057/103-1240-0057",
                  "train-other-960/103-1241-0019/103-1241-0019",
                  "train-other-960/103-1241-0025/103-1241-0025",
                  "train-other-960/103-1241-0043/103-1241-0043",
                  "train-other-960/1034-121119-0013/1034-121119-0013",
                ]
              ]
            if epoch in [60]:
              seq_tags_list += [["train-other-960/1246-124548-0042/1246-124548-0042"]]
            for seq_tags in seq_tags_list:
              recog.global_att_returnn_label_sync_beam_search(
                alias=train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
                corpus_keys=("train",),
                run_analysis=True,
                analyze_gradients=True,
                only_do_analysis=True,
                att_weight_seq_tags=seq_tags,
                analysis_ref_alignment_opts={
                  "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
                  "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
                  "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
                },
                analysis_analyze_gradients_plot_encoder_layers=True,
                analsis_analyze_gradients_plot_log_gradients=epoch in [10, 60],
                analysis_dump_self_att=epoch in [10, 20, 30, 40, 50, 60],
              )

          if (alias == "v8" and epoch in [11, 21, 31, 41, 51] + list(range(20, 31))):
            for seq_tags in [["train-other-960/40-222-0033/40-222-0033"]]:
              recog.global_att_returnn_label_sync_beam_search(
                alias=train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
                corpus_keys=("train",),
                run_analysis=True,
                analyze_gradients=True,
                only_do_analysis=True,
                att_weight_seq_tags=seq_tags,
                analysis_ref_alignment_opts={
                  "ref_alignment_hdf": LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
                  "ref_alignment_blank_idx": LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
                  "ref_alignment_vocab_path": LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
                },
                analysis_analyze_gradients_plot_encoder_layers=True,
                analsis_analyze_gradients_plot_log_gradients=False,
              )

  plot_flipped_cross_att_weight_evolution_v2(flipped_att_weights_evolution_epochs, flipped_att_weights_evolution)
  # plot_flipped_self_att_weight_evolution()
  # plot_flipped_vs_normal_cross_att_weights()
  # plot_gradients_wrt_different_layers()

  for alias in [
    "v3_wo-wf-wo-ctx-in-state_ctc", "v3_wo-wf-w-ctx-in-state_ctc", "v3_w-wf-w-ctx-in-state_ctc"
  ]:
    plot_diff_models(
      [analyze_gradients_jobs[alias]],
      alias=f"enc-11-grads/global-aed/{alias}",
      titles=None,  # titles,
      folder_name="log-prob-grads_wrt_enc-11_log-space",
      scale=1.0,
    )


def plot_flipped_cross_att_weight_evolution_v2(epochs, analyze_gradients_jobs_list):
  cross_att_hdfs = [
    Path(DelayedFormat(
      "{}/enc-layer-12/att_weights/att_weights.hdf",
      analyze_gradients_job.out_files["cross-att"]
    ).get()) for analyze_gradients_job in analyze_gradients_jobs_list
  ]
  targets_hdf = analyze_gradients_jobs_list[0].out_files["targets.hdf"]

  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=cross_att_hdfs,
    targets_hdf=targets_hdf,
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=LIBRISPEECH_GMM_WORD_ALIGNMENT.model_hyperparameters.blank_idx,
    ref_alignment_hdf=LIBRISPEECH_GMM_WORD_ALIGNMENT.alignment_paths["train"],
    json_vocab_path=LibrispeechBPE1056_LABELS.vocab_path,
    ctc_alignment_hdf=None,
    ref_alignment_json_vocab_path=LIBRISPEECH_GMM_WORD_ALIGNMENT.vocab_path,
    plot_w_cog=False,
    titles=[f"Epoch {epoch * 4 / 20}" for epoch in epochs],
    vmin=0.0,
    vmax=1.0,
    # titles=[f"Epoch {epoch * 4 // 20 if (epoch * 4 / 20).is_integer() else epoch * 4 / 20}" for epoch in epochs],
  )
  plot_att_weights_job.add_alias(f"flipped_cross_att_evolution_v2")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)


def plot_flipped_cross_att_weight_evolution():
  epochs = [
    10,
    40,
    45,
    50,
  ]
  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=[
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-{epoch}-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/cross-att/enc-layer-12/att_weights/att_weights.hdf") for epoch in epochs
    ],
    targets_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-50-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/output/targets.hdf"),
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=0,
    ref_alignment_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_hdf_align.hdf"),
    json_vocab_path=Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"),
    ctc_alignment_hdf=None,
    segment_whitelist=["train-other-960/40-222-0033/40-222-0033"],
    ref_alignment_json_vocab_path=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_vocab"),
    plot_w_cog=False,
    titles=[f"Epoch {epoch * 4 / 20}" for epoch in epochs],
    # titles=[f"Epoch {epoch * 4 // 20 if (epoch * 4 / 20).is_integer() else epoch * 4 / 20}" for epoch in epochs],
  )
  plot_att_weights_job.add_alias(f"flipped_cross_att_evolution")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)


def plot_flipped_vs_normal_cross_att_weights():
  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=[
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-60-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/cross-att/enc-layer-12/att_weights/att_weights.hdf"),
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4_ce-aux-4-8/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/cross-att/enc-layer-12/att_weights/att_weights.hdf"),
    ],
    targets_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4_ce-aux-4-8/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/output/targets.hdf"),
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=0,
    ref_alignment_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_hdf_align.hdf"),
    json_vocab_path=Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"),
    ctc_alignment_hdf=None,
    segment_whitelist=["train-other-960/40-222-0033/40-222-0033"],
    ref_alignment_json_vocab_path=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_vocab"),
    plot_w_cog=False,
    titles=["Reversed encoder", "Standard encoder"],
    # titles=[f"Epoch {epoch * 4 // 20 if (epoch * 4 / 20).is_integer() else epoch * 4 / 20}" for epoch in epochs],
  )
  plot_att_weights_job.add_alias(f"flipped_vs_normal_cross_att_weights")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)


def plot_gradients_wrt_different_layers():
  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=[
      # Path(
      #   f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-60-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/x_linear/log-prob-grads_wrt_x_linear_log-space/att_weights.hdf"),
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-60-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/enc-8/log-prob-grads_wrt_enc-8_log-space/att_weights.hdf"),
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-60-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/enc-9/log-prob-grads_wrt_enc-9_log-space/att_weights.hdf"),
    ],
    targets_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4_ce-aux-4-8/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/output/targets.hdf"),
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=0,
    ref_alignment_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_hdf_align.hdf"),
    json_vocab_path=Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"),
    ctc_alignment_hdf=None,
    segment_whitelist=["train-other-960/40-222-0033/40-222-0033"],
    ref_alignment_json_vocab_path=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_vocab"),
    plot_w_cog=False,
    titles=["$G_9$", "$G_{10}$"],
    # titles=[f"Epoch {epoch * 4 // 20 if (epoch * 4 / 20).is_integer() else epoch * 4 / 20}" for epoch in epochs],
  )
  plot_att_weights_job.add_alias(f"gradients_wrt_different_layers/9_10")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)

  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=[
      # Path(
      #   f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-60-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/x_linear/log-prob-grads_wrt_x_linear_log-space/att_weights.hdf"),
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-10-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/enc-0/log-prob-grads_wrt_enc-0_log-space/att_weights.hdf"),
    ],
    targets_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4_ce-aux-4-8/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/output/targets.hdf"),
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=0,
    ref_alignment_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_hdf_align.hdf"),
    json_vocab_path=Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"),
    ctc_alignment_hdf=None,
    segment_whitelist=["train-other-960/40-222-0033/40-222-0033"],
    ref_alignment_json_vocab_path=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_vocab"),
    plot_w_cog=False,
    titles=["$G_1$"],
    # titles=[f"Epoch {epoch * 4 // 20 if (epoch * 4 / 20).is_integer() else epoch * 4 / 20}" for epoch in epochs],
  )
  plot_att_weights_job.add_alias(f"gradients_wrt_different_layers/1")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)

  plot_att_weights_job = PlotAttentionWeightsJobV2(
    att_weight_hdf=[
      # Path(
      #   f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-60-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/x_linear/log-prob-grads_wrt_x_linear_log-space/att_weights.hdf"),
      Path(
        f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-10-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/work/x_linear/log-prob-grads_wrt_x_linear_log-space/att_weights.hdf"),
    ],
    targets_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_standard-conformer/train_from_scratch/500-ep_bs-15000_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4_ce-aux-4-8/returnn_decoding/epoch-61-checkpoint/no-lm/beam-size-12/train/analysis/analyze_gradients_ground-truth/40-222-0033/output/targets.hdf"),
    seg_starts_hdf=None,
    seg_lens_hdf=None,
    center_positions_hdf=None,
    target_blank_idx=None,
    ref_alignment_blank_idx=0,
    ref_alignment_hdf=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_hdf_align.hdf"),
    json_vocab_path=Path("/work/asr4/zeineldeen/setups-data/librispeech/2022-11-28--conformer-att/work/i6_core/text/label/subword_nmt/train/ReturnnTrainBpeJob.qhkNn2veTWkV/output/bpe.vocab"),
    ctc_alignment_hdf=None,
    segment_whitelist=["train-other-960/40-222-0033/40-222-0033"],
    ref_alignment_json_vocab_path=Path("/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_experiments/users/schmitt/alignment/alignment/GmmAlignmentToWordBoundariesJob.Me7asSFVFnO6/output/out_vocab"),
    plot_w_cog=False,
    titles=["$G_0$"],
  )
  plot_att_weights_job.add_alias(f"gradients_wrt_different_layers/0")
  tk.register_output(plot_att_weights_job.get_one_alias(), plot_att_weights_job.out_plot_dir)


def plot_flipped_self_att_weight_evolution():
  epochs = [10, 20, 30, 32, 34, 38, 40, 50]
  for head in range(8, 9):
    plot_self_att_weights_job = PlotSelfAttentionWeightsOverEpochsJob(
      att_weight_hdfs=[
        Path(
          f"/u/schmitt/experiments/03-09-24_aed_flipped_encoder/alias/models/ls_conformer/global_att/baseline_v1/baseline_rf/bpe1056/w-weight-feedback/w-att-ctx-in-state/nb-lstm/12-layer_512-dim_conformer-w-abs-pos/train_from_scratch/500-ep_bs-15000_mgpu-4_w-sp_curric_lr-dyn_lr_piecewise_linear_epoch-wise_v2_reg-v1_filter-data-312000.0_accum-4/returnn_decoding/epoch-{epoch}-checkpoint/no-lm/beam-size-12/train/analysis/dump_self_att/ground-truth/output/self-att-energies_head-{head}.hdf") for epoch in epochs
      ],
      epochs=epochs,
    )
    plot_self_att_weights_job.add_alias(f"flipped_self_att_evolution_head-{head}")
    tk.register_output(plot_self_att_weights_job.get_one_alias(), plot_self_att_weights_job.out_plot_dir)
