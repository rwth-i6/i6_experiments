from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog
)

from i6_core.returnn.training import PtCheckpoint
from sisyphus import Path


def run_exps():
  # baseline model for checking consistency of train and recog implementations
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(10,),
      const_lr_list=(1e-4,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5, 129),
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(100,),
      const_lr_list=(1e-4,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5, 129), use_att_ctx_in_state=False, use_weight_feedback=False,
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(100,),
      const_lr_list=(1e-4,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,)
  ):
    for max_shift, num_iterations in [(1, 1), (2, 1), (1, 2)]:
      for train_alias, checkpoint in train.train_center_window_att_viterbi_import_global_tf(
        alias=model_alias,
        config_builder=config_builder,
        n_epochs_list=(100,),
        const_lr_list=(1e-4,),
        alignment_augmentation_opts={"max_shift": max_shift, "num_iterations": num_iterations},
      ):
        recog.center_window_returnn_frame_wise_beam_search(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=checkpoint,
        )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_viterbi_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(500,),
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
      )
