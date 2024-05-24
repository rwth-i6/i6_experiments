from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.config_builder import get_center_window_att_config_builder_rf


def center_window_att_baseline_rf(
        win_size_list: Tuple[int, ...] = (5, 129),
        decoder_version: int = 1,
        use_weight_feedback: bool = True,
):
  for win_size in win_size_list:
    alias = f"{base_alias}/baseline_rf/win-size-{win_size}/{'w' if use_weight_feedback else 'wo'}-weight-feedback/decoder-version-{decoder_version if decoder_version else 1}"
    yield alias, get_center_window_att_config_builder_rf(
      win_size=win_size,
      label_decoder_version=decoder_version,
      blank_decoder_version=3,
      use_joint_model=False,
      use_weight_feedback=use_weight_feedback,
    )