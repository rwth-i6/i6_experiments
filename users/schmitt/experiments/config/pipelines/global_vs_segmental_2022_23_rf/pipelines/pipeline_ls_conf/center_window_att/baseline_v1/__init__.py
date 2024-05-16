from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v1 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog
)
from i6_experiments.users.schmitt.custom_load_params import load_missing_params


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(10,),
      time_rqmt=4,
      train_opts=dict(
        aux_loss_layers=None,
        accum_grad_multiple_step=2,
        optimizer={"class": "adam", "epsilon": 1e-8}
      )
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",)
      )
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        pure_torch=True,
      )

  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,), decoder_version=2
  ):
    for train_alias, checkpoint in train.train_center_window_att_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(10,),
      time_rqmt=4,
      train_opts=dict(
        aux_loss_layers=None,
        accum_grad_multiple_step=2,
        optimizer={"class": "adam", "epsilon": 1e-8}
      ),
      custom_missing_load_func=load_missing_params
    ):
      recog.center_window_returnn_frame_wise_beam_search(
        alias=train_alias,
        config_builder=config_builder,
        checkpoint=checkpoint,
        checkpoint_aliases=("last",),
        pure_torch=True,
      )
