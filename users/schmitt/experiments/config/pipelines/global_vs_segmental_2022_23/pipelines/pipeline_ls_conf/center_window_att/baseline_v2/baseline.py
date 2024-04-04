from typing import Tuple, Optional, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.base import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias


def center_window_att_baseline(
        win_size_list: Tuple[int, ...] = (5, 129),
):
  for win_size in win_size_list:
    alias = f"{base_alias}/baseline/win-size-%d" % (
      win_size
    )
    yield alias, get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
      length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
    )


def center_window_att_baseline_w_length_scale(
        win_size_list: Tuple[int, ...],
        length_scale_list: Tuple[float, ...]
):
  for win_size in win_size_list:
    for length_scale in length_scale_list:
      alias = f"{base_alias}/baseline_w-length-scale/win-size-%d_length-scale-%f" % (
        win_size, length_scale
      )
      yield alias, get_center_window_att_config_builder(
        win_size=win_size,
        use_weight_feedback=True,
        length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
        length_scale=length_scale
      )
