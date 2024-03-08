from typing import Tuple

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  default_import_model_name,
  get_center_window_att_config_builder,
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train_new import SegmentalTrainExperiment
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.recog_new import ReturnnSegmentalAttDecodingExperiment


def center_window_att_import_global_global_ctc_align_no_finetuning(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = f"{base_alias}/no_finetuning/random_init_length_model/win-size-%d" % (
      win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
    )

    recog_exp = ReturnnSegmentalAttDecodingExperiment(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      checkpoint_alias="best-4-avg",
      recog_opts={
        "search_corpus_key": "dev-other",
        "load_ignore_missing_vars": True,  # otherwise RETURNN will complain about missing length model params
      },
    )
    recog_exp.run_eval()
    recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model(
        win_size_list: Tuple[int, ...] = (128,),
):
  for win_size in win_size_list:
    alias = f"{base_alias}/no_finetuning/no_length_model/win-size-%d" % (
      win_size
    )
    config_builder = get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
      length_scale=0.0,
    )

    recog_exp = ReturnnSegmentalAttDecodingExperiment(
      alias=alias,
      config_builder=config_builder,
      checkpoint=external_checkpoints[default_import_model_name],
      checkpoint_alias="best-4-avg",
      recog_opts={
        "search_corpus_key": "dev-other",
        "load_ignore_missing_vars": True,  # otherwise RETURNN will complain about missing length model params
      },
    )
    recog_exp.run_eval()
    recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_no_finetuning_no_length_model_blank_penalty(
        win_size_list: Tuple[int, ...] = (128,)
):
  for win_size in win_size_list:
    for blank_penalty in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4):
      alias = f"{base_alias}/no_finetuning/no_length_model_blank_penalty/win-size-%d_penalty-%f" % (
        win_size, blank_penalty
      )
      config_builder = get_center_window_att_config_builder(
        win_size=win_size,
        use_weight_feedback=True,
        length_scale=0.0,
        blank_penalty=blank_penalty
      )

      recog_exp = ReturnnSegmentalAttDecodingExperiment(
        alias=alias,
        config_builder=config_builder,
        checkpoint=external_checkpoints[default_import_model_name],
        checkpoint_alias="best-4-avg",
        recog_opts={
          "search_corpus_key": "dev-other",
          "load_ignore_missing_vars": True,  # otherwise RETURNN will complain about missing length model params
        },
      )
      recog_exp.run_eval()
      recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_only_train_length_model(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        analysis_checkpoint_alias: str = "best-4-avg"
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = f"{base_alias}/no_finetuning/only_train_length_model/win-size-%d_%d-epochs_%f-const-lr" % (
          win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
          use_old_global_att_to_seg_att_maker=False,
        )

        train_exp = SegmentalTrainExperiment(
          config_builder=config_builder,
          alias=alias,
          num_epochs=n_epochs,
          train_opts={"only_train_length_model": True}
        )
        checkpoints, model_dir, learning_rates = train_exp.run_train()

        for checkpoint_alias in ("last", "best", "best-4-avg"):
          recog_exp = ReturnnSegmentalAttDecodingExperiment(
            alias=alias,
            config_builder=config_builder,
            checkpoint={
              "model_dir": model_dir,
              "learning_rates": learning_rates,
              "key": "dev_score_label_model/output_prob",
              "checkpoints": checkpoints,
              "n_epochs": n_epochs
            },
            checkpoint_alias=checkpoint_alias,
            recog_opts={"search_corpus_key": "dev-other"},
          )
          recog_exp.run_eval()
          if checkpoint_alias == analysis_checkpoint_alias:
            recog_exp.run_analysis()


def center_window_att_import_global_global_ctc_align_only_train_length_model_chunking(
        win_size_list: Tuple[int, ...] = (5, 129),
        n_epochs_list: Tuple[int, ...] = (10,),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        chunk_params_data_list: Tuple[Tuple[int, int], ...] = ((200000, 100000),),
        analysis_checkpoint_alias: str = "best-4-avg"
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

          alias = f"{base_alias}/no_finetuning/only_train_length_model_chunking/win-size-%d_%d-epochs_%f-const-lr/chunk-size-%d_chunk-step-%d" % (
            win_size, n_epochs, const_lr, chunk_size, chunk_step
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

          train_exp = SegmentalTrainExperiment(
            config_builder=config_builder,
            alias=alias,
            num_epochs=n_epochs,
            train_opts={
              "only_train_length_model": True,
              "chunking_opts": chunking_opts
            }
          )
          checkpoints, model_dir, learning_rates = train_exp.run_train()

          for checkpoint_alias in ("last", "best", "best-4-avg"):
            recog_exp = ReturnnSegmentalAttDecodingExperiment(
              alias=alias,
              config_builder=config_builder,
              checkpoint={
                "model_dir": model_dir,
                "learning_rates": learning_rates,
                "key": "dev_score_label_model/output_prob",
                "checkpoints": checkpoints,
                "n_epochs": n_epochs
              },
              checkpoint_alias=checkpoint_alias,
              recog_opts={
                "search_corpus_key": "dev-other"
              },
            )
            recog_exp.run_eval()
            if checkpoint_alias == analysis_checkpoint_alias:
              recog_exp.run_analysis()

          # checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
          #   alias=alias,
          #   config_builder=config_builder,
          #   train_opts={
          #     "num_epochs": n_epochs,
          #     "const_lr": const_lr,
          #     "only_train_length_model": True,
          #     "chunking_opts": chunking_opts,
          #   }
          # )
          #
          # returnn_recog_center_window_att_import_global(
          #   alias=alias,
          #   config_builder=config_builder,
          #   checkpoint=checkpoints[n_epochs],
          #   recog_opts={
          #     "corpus_key": "dev-other",
          #   },
          # )


# def center_window_att_import_global_global_ctc_align_only_train_length_model_use_label_model_state_only_non_blank_ctx(
#         win_size_list: Tuple[int, ...] = (5, 129),
#         n_epochs_list: Tuple[int, ...] = (10,),
#         const_lr_list: Tuple[float, ...] = (1e-4,),
# ):
#   for win_size in win_size_list:
#     for n_epochs in n_epochs_list:
#       for const_lr in const_lr_list:
#         alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/only_train_length_model_use_label_model_state_only_non_blank_ctx/win-size-%d_%d-epochs_%f-const-lr" % (
#           default_import_model_name, win_size, n_epochs, const_lr
#         )
#         config_builder = get_center_window_att_config_builder(
#           win_size=win_size,
#           use_weight_feedback=True,
#           length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
#           use_old_global_att_to_seg_att_maker=False,
#         )
#
#         checkpoints, model_dir, learning_rates = train_center_window_att_import_global(
#           alias=alias,
#           config_builder=config_builder,
#           train_opts={
#             "num_epochs": n_epochs,
#             "const_lr": const_lr,
#             "only_train_length_model": True,
#           }
#         )
#
#         returnn_recog_center_window_att_import_global(
#           alias=alias,
#           config_builder=config_builder,
#           checkpoint=checkpoints[n_epochs],
#           recog_opts={
#             "corpus_key": "dev-other",
#           },
#         )


# def center_window_att_import_global_global_ctc_align_only_train_length_model_use_label_model_state_only_non_blank_ctx_eos(
#         win_size_list: Tuple[int, ...] = (5, 129),
#         n_epochs_list: Tuple[int, ...] = (10,),
#         const_lr_list: Tuple[float, ...] = (1e-4,),
# ):
#   for win_size in win_size_list:
#     for n_epochs in n_epochs_list:
#       for const_lr in const_lr_list:
#         alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/no_finetuning/only_train_length_model_use_label_model_state_only_non_blank_ctx_eos/win-size-%d_%d-epochs_%f-const-lr" % (
#           default_import_model_name, win_size, n_epochs, const_lr
#         )
#         config_builder = get_center_window_att_config_builder(
#           win_size=win_size,
#           use_weight_feedback=True,
#           length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
#           use_old_global_att_to_seg_att_maker=False,
#           search_remove_eos=True,
#         )
#
#         align_targets = ctc_aligns.global_att_ctc_align.ctc_alignments_with_eos(
#           segment_paths=config_builder.dependencies.segment_paths,
#           blank_idx=config_builder.dependencies.model_hyperparameters.blank_idx,
#           eos_idx=config_builder.dependencies.model_hyperparameters.sos_idx
#         )
#
#         center_window_checkpoints, _, _ = train_center_window_att_import_global(
#           alias=alias,
#           config_builder=config_builder,
#           train_opts={
#             "num_epochs": n_epochs,
#             "const_lr": const_lr,
#             "only_train_length_model": True,
#             "align_targets": align_targets,
#           }
#         )
#
#         returnn_recog_center_window_att_import_global(
#           alias=alias,
#           config_builder=config_builder,
#           checkpoint=center_window_checkpoints[n_epochs],
#           recog_opts={
#             "corpus_key": "dev-other",
#           },
#         )
