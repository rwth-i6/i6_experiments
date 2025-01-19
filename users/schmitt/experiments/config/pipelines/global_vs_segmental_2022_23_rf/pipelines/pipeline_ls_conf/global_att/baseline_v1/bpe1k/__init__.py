import os

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1 import (
  baseline
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att import (
  train, recog
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints, default_import_model_name
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.phonemes.gmm_alignments import LIBRISPEECH_GMM_WORD_ALIGNMENT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.checkpoints import lm_checkpoints

from i6_core.returnn import PtCheckpoint

from sisyphus import Path

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

    # v2: same as v1, but use epoch-wise OCLR
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      keep_epochs=list(range(1, 240)) + [500],
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        corpus_keys=("dev-other",),
      )
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        corpus_keys=("dev-other",),
        run_analysis=True,
        analyze_gradients=True,
        checkpoint_aliases=("last",),
      )
      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in [22, 55, 60] or epoch in range(1, 60, 5):
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

        if epoch == 406:
          for input_layer_name in ["encoder_input", "frontend_input"]:
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              run_analysis=True,
              analysis_dump_gradients=True,
              only_do_analysis=True,
              corpus_keys=("train",),
              att_weight_seq_tags=None,
              analysis_dump_gradients_input_layer_name=input_layer_name,
            )

    # v3: same as v2, but filter out data > 19.5s
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      keep_epochs=list(range(1, 240)) + [500],
      filter_data_len=19.5 * 16_000,  # sample rate 16kHz
    ):
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
        corpus_keys=("dev-other",),
        run_analysis=True,
        analyze_gradients=True,
        checkpoint_aliases=("last",),
      )

      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in [31, 55, 60] or epoch in range(32, 55) or epoch in range(1, 60, 5):
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

    # v3, 900 epochs
    keep_epochs = list(range(90, 900, 90))
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=900,
      keep_epochs=keep_epochs,
      filter_data_len=19.5 * 16_000,  # sample rate 16kHz
      gpu_mem_rqmt=24,
      use_mgpu=False,
      ctc_aux_loss_layers=(4, 8),
      accum_grad_multiple_step=2,
      batch_size=35_000,
    ):
      recog.global_att_returnn_label_sync_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        corpus_keys=("dev-other",),
      )

      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in keep_epochs:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
          )

    # v4: same as v2, but filter out targets > 75
    for train_alias, checkpoint in train.train_global_att(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs=500,
      keep_epochs=list(range(1, 240)) + [500],
      filter_target_len=75,  # sample rate 16kHz
    ):
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
        corpus_keys=("dev-other",),
        run_analysis=True,
        analyze_gradients=True,
        checkpoint_aliases=("last",),
      )

      for epoch, chckpt in checkpoint["checkpoints"].items():
        if epoch in [60, 80, 116] or epoch in range(1, 60, 5) or epoch in range(50, 70):
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=chckpt,
            checkpoint_aliases=(f"epoch-{epoch}",),
            run_analysis=True,
            analyze_gradients=True,
            only_do_analysis=True,
          )

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
  ) in [
    ["v3_big", None, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24], # v3_big: same as v2, but on 24gb GPU with batch size 40k - not flipped
    ["v3_big_rand-1337", 1337, None, None, False, 12, 512, False, False, False, list(range(1, 240)), 24],  # v3_big_rand-1337 - flipped
    ["v3_big_rand-8264", 8264, None, None, False, 12, 512, False, False, False, [121, 131, 141], 24],  # v3_big_rand-8264 - flipped
    ["v3_big_rand-2160", 2160, None, None, False, 12, 512, False, False, False, [121, 131, 141], 24],  # v3_big_rand-2160 - flipped
    ["v5_big", None, 21, None, False, 12, 512, False, False, False, list(range(1, 240)), 24],  # v5_big: same as v3_big, but enable self attention only after 20 sub-epochs (1 full epoch) - not flipped
    ["v8_big", None, None, (4, 8), False, 12, 512, False, False, False, list(range(1, 141, 10)), 24],  # v8_big: same as v3_big, but use CTC aux loss - not flipped
  ]:
    for model_alias, config_builder in baseline.global_att_baseline_rf(
            use_weight_feedback=True,
            label_type="bpe1056",
            conformer_wo_final_layer_norm_per_layer=conformer_wo_final_layer_norm_per_layer,
            conformer_num_layers=conformer_num_layers,
            conformer_out_dim=conformer_out_dim,
            conformer_wo_convolution=conformer_wo_convolution,
            conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
            conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
    ):
      if gpu_mem_rqmt == 24:
        use_mgpu = False
        accum_grad_multiple_step = 2
      else:
        use_mgpu = True
        accum_grad_multiple_step = 4

      for train_alias, checkpoint in train.train_global_att(
              alias=model_alias,
              config_builder=config_builder,
              n_epochs=2_000,
              batch_size=30_000 if alias == "v9_big" else 35_000,
              keep_epochs=keep_epochs,
              gpu_mem_rqmt=gpu_mem_rqmt,
              accum_grad_multiple_step=accum_grad_multiple_step,
              use_mgpu=use_mgpu,
              use_torch_amp=False,
              filter_data_len=19.5 * 16_000,
              random_seed=random_seed,
              disable_enc_self_att_until_epoch=disable_self_att_until_epoch,
              ctc_aux_loss_layers=ctc_aux_loss_layers,
      ):
        if "rand" in alias:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            corpus_keys=("dev-other", "dev-clean", "test-other", "test-clean"),
          )
        else:
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            corpus_keys=("dev-other", "dev-clean", "test-other", "test-clean"),
            checkpoint_aliases=("last",)
          )

        if alias == "v5_big":
          for base_scale, ext_scale in [
            (1.0, 1.2),
            (0.1, 0.9),
            (0.2, 0.8),
            (0.3, 0.7),
            (0.4, 0.6),
            (0.5, 0.5),
            (0.6, 0.4),
            (0.7, 0.3),
            (0.8, 0.2),
            (0.9, 0.1),
            (0.1, 1.0),
            (0.2, 1.0),
            (0.3, 1.0),
            (0.4, 1.0),
            (0.5, 1.0),
            (0.6, 1.0),
            (0.7, 1.0),
            (0.8, 1.0),
            (0.9, 1.0),
            (1.1, 1.0),
            (1.2, 1.0),
            (1.3, 1.0),
            (1.0, 1.0),
          ]:
            corpus_keys = ["dev-other"]
            if base_scale == 1.0 and ext_scale == 1.0:
              corpus_keys += ["test-other"]
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=checkpoint,
              corpus_keys=corpus_keys,
              checkpoint_aliases=("last",),
              external_aed_opts={
                "checkpoint": PtCheckpoint(Path(
                  "/u/schmitt/experiments/03-09-24_aed_flipped_encoder/work/i6_core/returnn/training/ReturnnTrainingJob.czqZZvX66f4j/output/models/epoch.2000.pt"
                )),
                "scale": ext_scale,
              },
              base_scale=base_scale,
            )

        if alias in (
                # "v5_big",
                # "v8_big",
                "v3_big",
        ):
          for lm_scale, ilm_scale in [
            (0.54, 0.4),
            # (0.5, 0.4),
            # (0.6, 0.4),
          ]:
            corpus_keys = ["dev-other"]
            beam_size_list = [12]
            if lm_scale == 0.54 and ilm_scale == 0.4:
              corpus_keys = ["test-other"]
              beam_size_list = [12, 84]

            lm_alias = "1k_max-seq-length-112_24-layers_512-dim"
            # lm_alias = "1k_max-seq-length-112_24-layers_1024-dim"
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=checkpoint,
              corpus_keys=corpus_keys,
              checkpoint_aliases=("last",),
              lm_type="trafo",
              lm_scale_list=(lm_scale,),
              ilm_scale_list=(ilm_scale,),
              ilm_type="mini_att",
              lm_alias=lm_alias,
              lm_checkpoint=lm_checkpoints[lm_alias],
              behavior_version=21,  # otherwise, trafo lm logits has wrong weight order
              beam_size_list=beam_size_list,
              sbatch_args=["-p", "gpu_48gb,gpu_24gb_preemptive,gpu_11gb"],
            )

        recog.global_att_returnn_label_sync_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
          corpus_keys=("dev-other",),
          run_analysis=True,
          analyze_gradients=True,
          checkpoint_aliases=("last",),
        )

        if alias == "v8_big":
          recog.global_att_returnn_label_sync_beam_search(
            alias=train_alias,
            config_builder=config_builder,
            checkpoint=checkpoint,
            checkpoint_aliases=("last",),
            run_analysis=True,
            only_do_analysis=True,
            analyze_gradients=True,
            analsis_analyze_gradients_plot_log_gradients=True,
            analysis_analyze_gradients_plot_encoder_layers=True,
            att_weight_seq_tags=[
              "train-other-960/1246-124548-0042/1246-124548-0042",
              "train-other-960/40-222-0033/40-222-0033",
              "train-other-960/103-1240-0038/103-1240-0038",
            ],
            corpus_keys=("train",),
          )

        analysis_epochs = [121, 131]
        if alias in ("v3_big",):
          analysis_epochs += [1355]
        if alias in ("v3_big", "v3_big_rand-1337", "v5_big", "v6_big"):
          analysis_epochs += list(range(1, 141, 10))
        if alias in ("v3_big", "v3_big_rand-1337"):
          analysis_epochs += list(range(90, 141))
        if alias in ("v3_big_rand-8264",):
          analysis_epochs += list(range(118, 131)) + list(range(10, 111, 10))

        for epoch, chckpt in checkpoint["checkpoints"].items():
          if alias == "v3_big_rand-1337" and epoch == 130:
            analysis_do_forced_align_on_gradients = True
          else:
            analysis_do_forced_align_on_gradients = False

          if epoch in analysis_epochs:
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              run_analysis=True,
              analyze_gradients=True,
              only_do_analysis=True,
              analysis_do_forced_align_on_gradients=analysis_do_forced_align_on_gradients,
            )
          if alias == "v3_big" and epoch in range(51, 141, 10):
            recog.global_att_returnn_label_sync_beam_search(
              alias=train_alias,
              config_builder=config_builder,
              checkpoint=chckpt,
              checkpoint_aliases=(f"epoch-{epoch}",),
              run_analysis=True,
              analyze_gradients=True,
              only_do_analysis=True,
              analysis_plot_encoder_gradient_graph=True,
              att_weight_seq_tags=["dev-other/116-288047-0013/116-288047-0013"]
            )
          if (
                  alias == "v8_big" and epoch == 919) or (
                  alias == "v3_big_rand-1337" and epoch == 646) or (
                  alias == "v3_big_rand-1337" and epoch == 141) or (
                  alias == "v3_big" and epoch == 141) or (
          ):
            if (
                    alias == "v3_big_rand-1337" and epoch == 646) or (
                    alias == "v3_big_rand-1337" and epoch == 141) or (
                    alias == "v3_big" and epoch == 141
            ):
              input_layer_names = ["frontend_input"]
            else:
              input_layer_names = ["encoder_input", "frontend_input"]
            for input_layer_name in input_layer_names:
              recog.global_att_returnn_label_sync_beam_search(
                alias=train_alias,
                config_builder=config_builder,
                checkpoint=chckpt,
                checkpoint_aliases=(f"epoch-{epoch}",),
                run_analysis=True,
                analysis_dump_gradients=True,
                only_do_analysis=True,
                corpus_keys=("train",),
                att_weight_seq_tags=None,
                analysis_dump_gradients_input_layer_name=input_layer_name,
              )

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
      lr_scheduling_opts={"type": "dyn_lr_piecewise_linear"},
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
      lr_scheduling_opts={"type": "dyn_lr_piecewise_linear"},
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




