from typing import Tuple


from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  default_import_model_name,
  get_center_window_att_config_builder,
  standard_train_recog_center_window_att_import_global,
  recog_center_window_att_import_global,
  rasr_recog_center_window_att_import_global
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.checkpoints import external_checkpoints
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf import ctc_aligns

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.train import SegmentalTrainExperiment


def center_window_att_import_global_global_ctc_align_baseline(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/win-size-%d_%d-epochs_%f-const-lr" % (
          default_import_model_name, win_size, n_epochs, const_lr
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=True,
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_no_weight_feedback(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        use_old_global_att_to_seg_att_maker: bool = True,
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/no_weight_feedback/win-size-%d_%d-epochs_%f-const-lr/%s" % (
          default_import_model_name, win_size, n_epochs, const_lr, "old_seg_att_maker" if use_old_global_att_to_seg_att_maker else "new_seg_att_maker"
        )
        config_builder = get_center_window_att_config_builder(
          win_size=win_size,
          use_weight_feedback=False,
          use_old_global_att_to_seg_att_maker=use_old_global_att_to_seg_att_maker
        )

        standard_train_recog_center_window_att_import_global(
          config_builder=config_builder,
          alias=alias,
          n_epochs=n_epochs,
          const_lr=const_lr
        )


def center_window_att_import_global_global_ctc_align_only_train_length_model(
        win_size_list: Tuple[int, ...] = (2, 4, 8, 16, 32, 64, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/only_train_length_model/win-size-%d_%d-epochs_%f-const-lr" % (
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


def center_window_att_import_global_global_ctc_align_no_weight_feedback_rasr_recog(
        win_size_list: Tuple[int, ...] = (4, 128),
        n_epochs_list: Tuple[int, ...] = (10, 100),
        const_lr_list: Tuple[float, ...] = (1e-4,),
        max_segment_len_list: Tuple[int, ...] = (20,),
):
  for win_size in win_size_list:
    for n_epochs in n_epochs_list:
      for const_lr in const_lr_list:
        alias = "models/ls_conformer/import_%s/center-window_att_global_ctc_align/simple_ablations/rasr_recog/win-size-%d_%d-epochs_%f-const-lr" % (
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

        for max_segment_len in max_segment_len_list:
          rasr_recog_center_window_att_import_global(
            alias=alias,
            config_builder=config_builder,
            checkpoint=checkpoints[n_epochs],
            analyse=True,
            search_corpus_key="dev-other",
            search_rqmt={"mem": 8, "time": 12},
            max_segment_len=max_segment_len,
          )
