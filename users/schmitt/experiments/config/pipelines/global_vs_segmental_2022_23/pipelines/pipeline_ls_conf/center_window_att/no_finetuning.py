from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  default_import_model_name,
  get_center_window_att_config_builder,
  standard_train_recog_center_window_att_import_global,
  returnn_recog_center_window_att_import_global,
  train_center_window_att_import_global,
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_ROOT, RETURNN_EXE

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment

from i6_experiments.users.schmitt.alignment.alignment import AlignmentAddEosJob


def center_window_att_import_global_global_ctc_align_no_finetuning(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/random_init_length_model/win-size-%d" % (
      default_import_model_name, win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
    )

    returnn_recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      recog_opts={
        "analyse": False,
        "search_corpus_key": "dev-other",
        "load_ignore_missing_vars": True,  # otherwise RETURNN will complain about missing length model params
      },
    )


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/no_length_model/win-size-%d" % (
      default_import_model_name, win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
      length_scale=0.0,
    )

    returnn_recog_center_window_att_import_global(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      recog_opts={
        "analyse": False,
        "search_corpus_key": "dev-other",
        "load_ignore_missing_vars": True,  # otherwise RETURNN will complain about missing length model params
      },
    )


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model_blank_penalty(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    for blank_penalty in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4):
      alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/no_length_model_blank_penalty/win-size-%d_penalty-%f" % (
        default_import_model_name, win_size, blank_penalty
      )
      config_builder = get_center_window_att_config_builder(
        win_size=win_size,
        use_weight_feedback=True,
        length_scale=0.0,
        blank_penalty=blank_penalty
      )

      returnn_recog_center_window_att_import_global(
        alias=alias,
        config_builder=config_builder,
        checkpoint=external_checkpoints[default_import_model_name],
        recog_opts={
          "analyse": False,
          "search_corpus_key": "dev-other",
          "load_ignore_missing_vars": True,  # otherwise RETURNN will complain about missing length model params
        },
      )


def center_window_att_import_global_global_ctc_align_only_train_length_model(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/only_train_length_model/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          use_old_global_att_to_seg_att_maker=True,
        )

        checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          train_opts={
            "num_epochs": n_epochs,
            "const_lr": const_lr,
            "only_train_length_model": True,
          }
        )

        returnn_recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          recog_opts={
            "analyse": False,
            "search_corpus_key": "dev-other",
          },
        )


def center_window_att_import_global_global_ctc_align_only_train_length_model_chunking(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        chunk_params_data_list: Tuple[Tuple[int, int], ...] = ((200000, 100000),),
):
  chunking_opts = {
    "chunk_size_targets": 0,
    "chunk_step_targets": 0,
    "red_factor": 960,
    "red_subtrahend": 399,
  }

  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        for chunk_params_data in chunk_params_data_list:
          chunk_size, chunk_step = chunk_params_data

          alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/only_train_length_model_chunking/win-size-%d_%d-epochs_%f-const-lr/chunk-size-%d_chunk-step-%d" % (
            default_import_model_name, win_size, n_epochs, const_lr, chunk_size, chunk_step
          )

          chunking_opts.update({
            "chunk_size_data": chunk_size,
            "chunk_step_data": chunk_step,
          })

          config_builder = get_center_window_att_config_builder(
            win_size=win_size,
            use_weight_feedback=True,
            use_old_global_att_to_seg_att_maker=False,
          )

          checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            train_opts={
              "num_epochs": n_epochs,
              "const_lr": const_lr,
              "only_train_length_model": True,
              "chunking_opts": chunking_opts,
            }
          )

          returnn_recog_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            checkpoint=checkpoints[n_epochs],
            recog_opts={
              "analyse": True,
              "search_corpus_key": "dev-other",
            },
          )


def center_window_att_import_global_global_ctc_align_only_train_length_model_use_label_model_state_only_non_blank_ctx(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/only_train_length_model_use_label_model_state_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False,
        )

        checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          train_opts={
            "num_epochs": n_epochs,
            "const_lr": const_lr,
            "only_train_length_model": True,
          }
        )

        returnn_recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=checkpoints[n_epochs],
          recog_opts={
            "analyse": True,
            "search_corpus_key": "dev-other",
          },
        )


def center_window_att_import_global_global_ctc_align_only_train_length_model_use_label_model_state_only_non_blank_ctx_eos(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/only_train_length_model_use_label_model_state_only_non_blank_ctx_eos/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
          use_old_global_att_to_seg_att_maker=False,
          search_remove_eos=True,
        )

        align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments_with_eos(
          segment_paths=config_builder.dependencies.segment_paths,
          blank_idx=config_builder.dependencies.model_hyperparameters.blank_idx,
          eos_idx=config_builder.dependencies.model_hyperparameters.sos_idx
        )

        center_window_checkpoints, _, _ = train_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          train_opts={
            "num_epochs": n_epochs,
            "const_lr": const_lr,
            "only_train_length_model": True,
            "align_targets": align_targets,
          }
        )

        returnn_recog_center_window_att_import_global(
          alias=alias,
          config_builder=config_builder,
          checkpoint=center_window_checkpoints[n_epochs],
          recog_opts={
            "analyse": True,
            "search_corpus_key": "dev-other",
          },
        )
