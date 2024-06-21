from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.config_builder import get_center_window_att_config_builder_rf


def center_window_att_baseline_rf(
        win_size_list: Tuple[int, ...] = (5, 129),
        use_att_ctx_in_state: bool = True,
        use_weight_feedback: bool = True,
        blank_decoder_version: int = 3,
        label_decoder_state: str = "nb-lstm",
):
  for win_size in win_size_list:
    alias, config_builder = get_center_window_att_config_builder_rf(
      win_size=win_size,
      use_att_ctx_in_state=use_att_ctx_in_state,
      blank_decoder_version=blank_decoder_version,
      use_joint_model=False,
      use_weight_feedback=use_weight_feedback,
      label_decoder_state=label_decoder_state,
    )
    alias = f"{base_alias}/baseline_rf/{alias}"
    yield alias, config_builder
