from typing import Tuple, Optional, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.baseline import (
  get_center_window_att_config_builder,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.center_window_att.baseline_v2.alias import alias as base_alias


def center_window_att_decoder_variation(
        win_size_list: Tuple[int, ...],
        decoder_version: int,
):
  for win_size in win_size_list:
    alias = f"{base_alias}/decoder-variations/v%d/win-size-%d" % (
      decoder_version, win_size
    )
    yield alias, get_center_window_att_config_builder(
      win_size=win_size,
      use_weight_feedback=True,
      length_model_opts={"use_label_model_state": True, "use_alignment_ctx": False},
      decoder_version=decoder_version,
    )
