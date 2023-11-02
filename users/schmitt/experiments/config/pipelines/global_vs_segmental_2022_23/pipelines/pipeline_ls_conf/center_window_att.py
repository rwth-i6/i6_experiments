from typing import Dict, Optional, List, Any, Tuple, Union
import copy

from sisyphus import Path

from i6_core.returnn.training import AverageTFCheckpointsJob, GetBestEpochJob, Checkpoint

# from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.segmental import LibrispeechConformerSegmentalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog import ReturnnDecodingExperimentV2


default_import_model_name = "glob.conformer.mohammad.5.6"


def center_window_att_import_global_global_ctc_align_no_finetuning(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_no_finetuning/win-size-%d" % (
      default_import_model_name, win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
    )

    recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      analyse=False,
      search_corpus_key="dev-other",
      load_ignore_missing_vars=True,  # otherwise RETURNN will complain about missing length model params
    )


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_no_finetuning_no_length_model/win-size-%d" % (
      default_import_model_name, win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
      length_scale=0.0,
    )

    recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      analyse=False,
      search_corpus_key="dev-other",
      load_ignore_missing_vars=True,  # otherwise RETURNN will complain about missing length model params
    )


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model_blank_penalty(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    for blank_penalty in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4):
      alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_no_finetuning_no_length_model_blank_penalty/win-size-%d_penalty-%f" % (
        default_import_model_name, win_size, blank_penalty
      )
      config_builder = get_center_window_att_config_builder(
        win_size=win_size,
        use_weight_feedback=True,
        length_scale=0.0,
        blank_penalty=blank_penalty
      )

      recog_center_window_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=external_checkpoints[default_import_model_name],
        analyse=False,
        search_corpus_key="dev-other",
        load_ignore_missing_vars=True,  # otherwise RETURNN will complain about missing length model params
      )


def center_window_att_import_global_global_ctc_align(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          }
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other"
        )


def center_window_att_import_global_global_ctc_align_length_model_no_label_feedback(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_length_model_no_label_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
          length_model_opts={"use_embedding": False}
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          }
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other",
        )


def center_window_att_import_global_global_ctc_align_length_model_diff_emb_size(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        emb_size_list: Tuple[int, ...] = (64,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for emb_size in emb_size_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_length_model_diff_emb_size/win-size-%d_%d-epochs_%f-const-lr/emb-size-%d" % (
            default_import_model_name, win_size, n_epochs, const_lr, emb_size
          )
          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=False,
            length_model_opts={"use_embedding": True, "embedding_size": emb_size}
          )

          train_exp = SegmentalTrainExperiment(
            config_builder=config_builder,
            alias=alias,
            n_epochs=n_epochs,
            import_model_train_epoch1=external_checkpoints[default_import_model_name],
            align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
            lr_opts={
              "type": "const_then_linear",
              "const_lr": const_lr,
              "const_frac": 1 / 3,
              "final_lr": 1e-6,
              "num_epochs": n_epochs
            }
          )
          checkpoints, model_dir, learning_rates = train_exp.run_train()

          recog_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            checkpoint=checkpoints[n_epochs],
            analyse=True,
            search_corpus_key="dev-other"
          )


def center_window_att_import_global_global_ctc_align_att_weight_penalty_recog(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for mult_weight in (
                0.005, #0.01, 0.02, 0.03, 0.04, 0.05
        ):
          for exp_weight in (2.0,):
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_att_weight_penalty_recog/win-size-%d_%d-epochs_%f-const-lr/mult-weight-%f_exp-weight-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, mult_weight, exp_weight
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
            )
            train_exp = SegmentalTrainExperiment(
              config_builder=train_config_builder,
              alias=alias,
              n_epochs=n_epochs,
              import_model_train_epoch1=external_checkpoints[default_import_model_name],
              align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
              lr_opts={
                "type": "const_then_linear",
                "const_lr": const_lr,
                "const_frac": 1 / 3,
                "final_lr": 1e-6,
                "num_epochs": n_epochs
              },
            )
            checkpoints, model_dir, learning_rates = train_exp.run_train()

            w_penalty_alias = alias + "/w_penalty"
            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              att_weight_recog_penalty_opts={
                "mult_weight": mult_weight,
                "exp_weight": exp_weight
              },
            )
            recog_center_window_att_import_global(
              alias=w_penalty_alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              analyse=True,
              search_corpus_key="dev-other",
            )

            wo_penalty_alias = alias + "/wo_penalty"
            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              att_weight_recog_penalty_opts=None,
            )
            recog_center_window_att_import_global(
              alias=wo_penalty_alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              analyse=True,
              search_corpus_key="dev-other",
            )


def center_window_att_import_global_global_ctc_align_att_weight_penalty_train(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        mult_weight_list: Tuple[float, ...] = (0.005,),
        exp_weight_list: Tuple[float, ...] = (2.0,),
        loss_scale_list: Tuple[float, ...] = (1.0,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for mult_weight in mult_weight_list:
          for exp_weight in exp_weight_list:
            for loss_scale in loss_scale_list:
              alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_att_weight_penalty_train/win-size-%d_%d-epochs_%f-const-lr/mult-weight-%f_exp-weight-%f/loss-scale-%f" % (
                default_import_model_name, win_size, n_epochs, const_lr, mult_weight, exp_weight, loss_scale
              )

              train_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                att_weight_recog_penalty_opts={
                  "mult_weight": mult_weight,
                  "exp_weight": exp_weight,
                  "use_as_loss": True,
                  "loss_scale": loss_scale,
                },
              )
              train_exp = SegmentalTrainExperiment(
                config_builder=train_config_builder,
                alias=alias,
                n_epochs=n_epochs,
                import_model_train_epoch1=external_checkpoints[default_import_model_name],
                align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
                lr_opts={
                  "type": "const_then_linear",
                  "const_lr": const_lr,
                  "const_frac": 1 / 3,
                  "final_lr": 1e-6,
                  "num_epochs": n_epochs
                },
              )
              checkpoints, model_dir, learning_rates = train_exp.run_train()

              w_penalty_alias = alias + "/w_penalty"
              recog_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                att_weight_recog_penalty_opts={
                  "mult_weight": mult_weight,
                  "exp_weight": exp_weight
                },
              )
              recog_center_window_att_import_global(
                alias=w_penalty_alias,
                config_builder=recog_config_builder,
                checkpoint=checkpoints[n_epochs],
                analyse=True,
                search_corpus_key="dev-other",
              )

              wo_penalty_alias = alias + "/wo_penalty"
              recog_config_builder = get_center_window_att_config_builder(
                win_size=win_size,
                use_weight_feedback=True,
                att_weight_recog_penalty_opts=None,
              )
              recog_center_window_att_import_global(
                alias=wo_penalty_alias,
                config_builder=recog_config_builder,
                checkpoint=checkpoints[n_epochs],
                analyse=True,
                search_corpus_key="dev-other",
              )


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_gaussian_att_weight_interpolation/win-size-%d_%d-epochs_%f-const-lr/std-%f_gauss_scale-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, std, gauss_scale
            )

            config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale}
            )
            train_exp = SegmentalTrainExperiment(
              config_builder=config_builder,
              alias=alias,
              n_epochs=n_epochs,
              import_model_train_epoch1=external_checkpoints[default_import_model_name],
              align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
              lr_opts={
                "type": "const_then_linear",
                "const_lr": const_lr,
                "const_frac": 1 / 3,
                "final_lr": 1e-6,
                "num_epochs": n_epochs
              },
            )
            checkpoints, model_dir, learning_rates = train_exp.run_train()

            recog_center_window_att_import_global(
              alias=alias,
              config_builder=config_builder,
              checkpoint=checkpoints[n_epochs],
              analyse=True,
              search_corpus_key="dev-other",
            )


def center_window_att_import_global_global_ctc_align_gaussian_att_weight_interpolation_plus_att_weight_recog_penalty(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        std_list: Tuple[float, ...] = (1.,),
        gauss_scale_list: Tuple[float, ...] = (.5,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for std in std_list:
          for gauss_scale in gauss_scale_list:
            alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_gaussian_att_weight_interpolation_plus_att_weight_recog_penalty/win-size-%d_%d-epochs_%f-const-lr/std-%f_gauss_scale-%f" % (
              default_import_model_name, win_size, n_epochs, const_lr, std, gauss_scale
            )

            train_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              gaussian_att_weight_interpolation_opts={"std": std, "gauss_scale": gauss_scale}
            )
            train_exp = SegmentalTrainExperiment(
              config_builder=train_config_builder,
              alias=alias,
              n_epochs=n_epochs,
              import_model_train_epoch1=external_checkpoints[default_import_model_name],
              align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
              lr_opts={
                "type": "const_then_linear",
                "const_lr": const_lr,
                "const_frac": 1 / 3,
                "final_lr": 1e-6,
                "num_epochs": n_epochs
              },
            )
            checkpoints, model_dir, learning_rates = train_exp.run_train()

            recog_config_builder = get_center_window_att_config_builder(
              win_size=win_size,
              use_weight_feedback=True,
              att_weight_recog_penalty_opts={
                "mult_weight": 0.005,
                "exp_weight": 2.0
              },
            )
            recog_center_window_att_import_global(
              alias=alias,
              config_builder=recog_config_builder,
              checkpoint=checkpoints[n_epochs],
              analyse=True,
              search_corpus_key="dev-other",
            )


def center_window_att_import_global_global_ctc_align_expected_position_aux_loss(
        win_size_list: Tuple[int, ...] = (128,),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        loss_scale_list: Tuple[float, ...] = (1.,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for loss_scale in loss_scale_list:
          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_expected_position_aux_loss/win-size-%d_%d-epochs_%f-const-lr/loss_scale-%f" % (
            default_import_model_name, win_size, n_epochs, const_lr, loss_scale
          )

          train_config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            expected_position_aux_loss_opts={"loss_scale": loss_scale},
            length_model_opts={"use_embedding": False},
          )
          train_exp = SegmentalTrainExperiment(
            config_builder=train_config_builder,
            alias=alias,
            n_epochs=n_epochs,
            import_model_train_epoch1=external_checkpoints[default_import_model_name],
            align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
            lr_opts={
              "type": "const_then_linear",
              "const_lr": const_lr,
              "const_frac": 1 / 3,
              "final_lr": 1e-6,
              "num_epochs": n_epochs
            },
          )
          checkpoints, model_dir, learning_rates = train_exp.run_train()


def center_window_att_import_global_global_ctc_align_chunking(
        win_size_list: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  chunking_opts = {
    "chunk_size_targets": 60,
    "chunk_step_targets": 30,
    "chunk_size_data": 360,
    "chunk_step_data": 180
  }

  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_chunking/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          chunking_opts=chunking_opts,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other"
        )


def center_window_att_import_global_global_ctc_align_weight_feedback(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_weight_feedback/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other",
          att_weight_seq_tags=[
            "dev-other/6467-94831-0006/6467-94831-0006",  # global 2 err, win-size-8 + win-size-128 + seg correct
            "dev-other/8254-84205-0021/8254-84205-0021",  # seg + win-size-8 + win-size-128 2 err, global correct
            "dev-other/6123-59150-0002/6123-59150-0002",  # seg + win-size-8 2 err, win-size-128 1 err, global correct
            "dev-other/1585-131718-0027/1585-131718-0027",  # global 2 err, win-size-8 + win-size-128 + seg correct
            "dev-other/1585-157660-0007/1585-157660-0007",  # seg 5 err, win-size-8 + win-size-128 4 err, global correct
            "dev-other/6123-59150-0008/6123-59150-0008",  # global 2 err, win-size-8 + win-size-128 1 err, seg correct
            "dev-other/1650-167613-0026/1650-167613-0026",  # seg 2 err, win-size-8 + win-size-128 3 err, global correct
            "dev-other/1686-142278-0018/1686-142278-0018",  # global 2 err, win-size-8 + win-size-128 + seg correct
            "dev-other/1701-141759-0026/1701-141759-0026",  # seg + win-size-8 + win-size-128 2 err, global correct
            "dev-other/2506-11278-0017/2506-11278-0017",  # all correct
            "dev-other/2506-11278-0025/2506-11278-0025",  # all correct
            "dev-other/2506-13150-0004/2506-13150-0004",  # all correct
            "dev-other/3660-172182-0035/3660-172182-0035",  # seg + win-size-8 + win-size-128 2 err, global correct
            "dev-other/4153-186222-0014/4153-186222-0014",  # global 3 err, win-size-8 + win-size-128 1 err, seg correct
            "dev-other/4570-14911-0000/4570-14911-0000",  # global 2 err, win-size-8 + win-size-128 1 err, seg correct
            "dev-other/5849-50873-0033/5849-50873-0033",  # seg 2 err, global + win-size-8 + win-size-128 correct
            "dev-other/6123-59186-0009/6123-59186-0009",  # seg 1 err, win-size-8 1 err, global + win-size-128 correct
            "dev-other/6267-65525-0049/6267-65525-0049",  # global 2 err, win-size-8 + win-size-128 + seg correct
            "dev-other/8288-274162-0025/8288-274162-0025",  # global 3 err, win-size-8 + win-size-128 + seg correct
          ],
        )


def center_window_att_import_global_global_ctc_align_large_window_problems():
  win_size_list = (4, 128)
  n_epochs = 10
  const_lr = 1e-4

  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_large_window_problems/win-size-%d_%d-epochs_%f-const-lr" % (
      default_import_model_name, win_size, n_epochs, const_lr
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
    )

    train_exp = SegmentalTrainExperiment(
      config_builder=config_builder,
      alias=alias,
      n_epochs=n_epochs,
      import_model_train_epoch1=external_checkpoints[default_import_model_name],
      align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
      lr_opts={
        "type": "const_then_linear",
        "const_lr": const_lr,
        "const_frac": 1 / 3,
        "final_lr": 1e-6,
        "num_epochs": n_epochs
      },
    )
    checkpoints, model_dir, learning_rates = train_exp.run_train()

    recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=checkpoints[n_epochs],
      analyse=True,
      search_corpus_key="dev-other",
    )


def center_window_att_import_global_global_ctc_align_positional_embedding(
        win_size_list: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_positional_embedding/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
          use_positional_embedding=True
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          }
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other"
        )


def center_window_att_import_global_global_ctc_align_only_train_length_model(
        win_size_list: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align_only_train_length_model/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          import_model_train_epoch1=external_checkpoints[default_import_model_name],
          align_targets=ctc_aligns.global_att_ctc_align.ctc_alignments,
          only_train_length_model=True,
          lr_opts={
            "type": "const_then_linear",
            "const_lr": const_lr,
            "const_frac": 1 / 3,
            "final_lr": 1e-6,
            "num_epochs": n_epochs
          },
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          analyse=True,
          search_corpus_key="dev-other"
        )


def get_center_window_att_config_builder(
        win_size: int,
        use_weight_feedback: bool = False,
        use_positional_embedding: bool = False,
        att_weight_recog_penalty_opts: Optional[Dict] = None,
        length_model_opts: Optional[Dict] = None,
        length_scale: float = 1.0,
        blank_penalty: float = 0.0,
        gaussian_att_weight_interpolation_opts: Optional[Dict] = None,
        expected_position_aux_loss_opts: Optional[Dict] = None,
        pos_pred_att_weight_interpolation_opts: Optional[Dict] = None,
):
  model_type = "librispeech_conformer_seg_att"
  variant_name = "seg.conformer.like-global"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["segment_center_window_size"] = win_size
  variant_params["network"]["use_weight_feedback"] = use_weight_feedback
  variant_params["network"]["use_positional_embedding"] = use_positional_embedding
  variant_params["network"]["att_weight_recog_penalty_opts"] = att_weight_recog_penalty_opts
  variant_params["network"]["gaussian_att_weight_interpolation_opts"] = gaussian_att_weight_interpolation_opts
  variant_params["network"]["pos_pred_att_weight_interpolation_opts"] = pos_pred_att_weight_interpolation_opts
  variant_params["network"]["expected_position_aux_loss_opts"] = expected_position_aux_loss_opts
  variant_params["network"]["length_scale"] = length_scale
  variant_params["network"]["blank_penalty"] = blank_penalty

  if length_model_opts:
    variant_params["network"]["length_model_opts"] = copy.deepcopy(length_model_opts)

  config_builder = LibrispeechConformerSegmentalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


# def train_center_window_att_import_global(
#         alias: str,
#         config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
#         align_targets: Dict[str, Path],
#         n_epochs: int,
#         import_model_name: str,
#         const_lr: float = 1e-4,
#         const_frac: float = 1/3,
#         final_lr: float = 1e-6,
#         cleanup_old_models: Optional[Dict] = None,
#         chunking_opts: Optional[Dict] = None,
#         only_train_length_model: bool = False,
# ):
#   cleanup_old_models = cleanup_old_models if cleanup_old_models is not None else {"keep_best_n": 1, "keep_last_n": 1}
#
#   train_opts = {
#     "cleanup_old_models": cleanup_old_models,
#     "lr_opts": {
#       "type": "const_then_linear",
#       "const_lr": const_lr,
#       "const_frac": const_frac,
#       "final_lr": final_lr,
#       "num_epochs": n_epochs
#     },
#     "import_model_train_epoch1": external_checkpoints[import_model_name],
#     "dataset_opts": {
#       "hdf_targets": align_targets
#     },
#     "only_train_length_model": only_train_length_model
#   }
#
#   if chunking_opts is not None:
#     train_opts["chunking"] = chunking_opts
#
#   checkpoints, model_dir, learning_rates = run_train(
#     config_builder=config_builder,
#     variant_params=config_builder.variant_params,
#     n_epochs=n_epochs,
#     train_opts=train_opts,
#     alias=alias
#   )
#
#   return checkpoints, model_dir, learning_rates


def recog_center_window_att_import_global(
        alias: str,
        config_builder: LibrispeechConformerSegmentalAttentionConfigBuilder,
        checkpoint: Checkpoint,
        search_corpus_key: str,
        concat_num: Optional[int] = None,
        search_rqmt: Optional[Dict[str, Any]] = None,
        batch_size: Optional[int] = None,
        analyse: bool = False,
        att_weight_seq_tags: Optional[List[str]] = None,
        load_ignore_missing_vars: bool = False,
):
  recog_exp = ReturnnDecodingExperimentV2(
    alias=alias,
    config_builder=config_builder,
    checkpoint=checkpoint,
    corpus_key=search_corpus_key,
    concat_num=concat_num,
    search_rqmt=search_rqmt,
    batch_size=batch_size,
    load_ignore_missing_vars=load_ignore_missing_vars,
  )
  recog_exp.run_eval()

  if analyse:
    if concat_num is not None:
      raise NotImplementedError

    recog_exp.run_analysis(
      ground_truth_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_hdf=ctc_aligns.global_att_ctc_align.ctc_alignments[search_corpus_key],
      att_weight_ref_alignment_blank_idx=10025,
      att_weight_seq_tags=att_weight_seq_tags,
    )


# def train_recog_center_window_att_import_global(
#         alias: str,
#         align_targets: Dict[str, Path],
#         n_epochs: int,
#         win_size: int,
#         import_model_name: str,
#         const_lr: float = 1e-4,
#         const_frac: float = 1/3,
#         final_lr: float = 1e-6,
#         cleanup_old_models: Optional[Dict] = None,
#         align_augment: bool = False,
#         use_weight_feedback: bool = False,
#         chunking_opts: Optional[Dict] = None,
#         use_positional_embedding: bool = False,
#         only_train_length_model: bool = False,
#         analyse: bool = False
# ):
#   config_builder = get_center_window_att_config_builder(
#     win_size=win_size,
#     use_weight_feedback=use_weight_feedback,
#     use_positional_embedding=use_positional_embedding
#   )
#
#   train_exp = SegmentalTrainExperiment(
#     config_builder=config_builder,
#     alias=alias,
#     n_epochs=n_epochs,
#     import_model_name=import_model_name,
#     const_lr=const_lr,
#     const_frac=const_frac,
#     final_lr=final_lr,
#     cleanup_old_models=cleanup_old_models,
#     align_augment=align_augment,
#     align_targets=align_targets,
#     chunking_opts=chunking_opts,
#     only_train_length_model=only_train_length_model
#   )
#
#   checkpoints, model_dir, learning_rates = train_exp.run_train()
#
#   recog_center_window_att_import_global(
#     alias=alias,
#     config_builder=config_builder,
#     checkpoint=checkpoints[n_epochs],
#     analyse=analyse,
#     search_corpus_key="dev-other"
#   )
#
#   return checkpoints
