from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v1 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog
)

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model import MakeModel
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_import import map_param_func_v2

from i6_experiments.users.zeyer.returnn.convert_ckpt_rf import ConvertTfCheckpointToRfPtJob

from i6_core.returnn.training import PtCheckpoint, Checkpoint

from sisyphus import Path


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    # for train_alias, checkpoint in train.train_center_window_att_from_scratch(
    #   alias=model_alias,
    #   config_builder=config_builder,
    #   n_epochs_list=(1,),
    #   time_rqmt=4
    # ):
    #   recog.center_window_returnn_frame_wise_beam_search(
    #     alias=train_alias,
    #     config_builder=config_builder,
    #     checkpoint=checkpoint,
    #     checkpoint_aliases=("last",)
    #   )

    for train_alias, checkpoint in train.train_center_window_att_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(1,),
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

    for train_alias, checkpoint in train.train_center_window_att_import_global_tf(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(20,),
      time_rqmt=12,
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

    # for train_alias, checkpoint in train.train_center_window_att_import_center_window_tf(
    #   alias=model_alias,
    #   config_builder=config_builder,
    #   n_epochs_list=(9,),
    #   time_rqmt=4
    # ):
    #   recog.center_window_returnn_frame_wise_beam_search(
    #     alias=train_alias,
    #     config_builder=config_builder,
    #     checkpoint=checkpoint,
    #     checkpoint_aliases=("last",)
    #   )
