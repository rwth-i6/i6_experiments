from typing import Tuple, Optional, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v5.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.config_builder import get_center_window_att_config_builder_rf


def center_window_att_baseline_rf(
        win_size_list: Tuple[int, ...] = (5, 129),
        use_att_ctx_in_state: bool = True,
        use_weight_feedback: bool = True,
        blank_decoder_version: int = 3,
        blank_decoder_opts: Optional[Dict] = None,
        bpe_vocab_size: int = 10025,
        use_current_frame_in_readout: bool = False,
        use_current_frame_in_readout_w_gate: bool = False,
        use_current_frame_in_readout_random: bool = False,
        use_correct_dim_tags: bool = False,
        use_trafo_att: bool = False,
        behavior_version: Optional[int] = None,
        use_trafo_att_wo_cross_att: bool = False,
):
  for win_size in win_size_list:
    alias, config_builder = get_center_window_att_config_builder_rf(
      win_size=win_size,
      use_att_ctx_in_state=use_att_ctx_in_state,
      blank_decoder_version=blank_decoder_version,
      blank_decoder_opts=blank_decoder_opts,
      use_joint_model=False,
      use_weight_feedback=use_weight_feedback,
      label_decoder_state="nb-2linear-ctx1",
      use_current_frame_in_readout=use_current_frame_in_readout,
      use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
      use_current_frame_in_readout_random=use_current_frame_in_readout_random,
      bpe_vocab_size=bpe_vocab_size,
      use_correct_dim_tags=use_correct_dim_tags,
      use_trafo_att=use_trafo_att,
      behavior_version=behavior_version,
      use_trafo_att_wo_cross_att=use_trafo_att_wo_cross_att,
    )
    alias = f"{base_alias}/baseline_rf/{alias}"
    yield alias, config_builder
