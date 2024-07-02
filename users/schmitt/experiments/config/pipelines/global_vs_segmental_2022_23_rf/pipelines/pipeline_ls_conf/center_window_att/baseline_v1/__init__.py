from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v1 import (
  baseline,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att import (
  train, recog, realign
)


def run_exps():
  for model_alias, config_builder in baseline.center_window_att_baseline_rf(
    win_size_list=(5,),
  ):
    for train_alias, checkpoint in train.train_center_window_att_full_sum_from_scratch(
      alias=model_alias,
      config_builder=config_builder,
      n_epochs_list=(125,),
      use_speed_pert=True,
      batch_size=15_000,
      time_rqmt=80,
      use_mgpu=False,
      beam_size=1,
      lattice_downsampling=1,
      alignment_interpolation_factor=0.0,
      train_on_viterbi_paths=True,
      only_use_blank_model=True,
    ):
      for epoch, chckpt in checkpoint["checkpoints"].items():
        realign.center_window_returnn_realignment(
          alias=train_alias,
          config_builder=config_builder,
          checkpoint=chckpt,
          checkpoint_alias=f"epoch-{epoch}",
          plot=True,
        )
