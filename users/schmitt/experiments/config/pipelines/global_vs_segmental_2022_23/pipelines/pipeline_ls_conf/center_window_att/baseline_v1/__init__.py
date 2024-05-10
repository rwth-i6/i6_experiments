from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v1 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att import (
  recog, train
)


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(1,),
      use_ctc_loss=False,
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
      )

    for train_alias, checkpoint in train.train_center_window_att_import_global_global_ctc_align(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(20,),
      use_ctc_loss=False,
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
      )



