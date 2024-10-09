from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v3.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.config_builder import get_center_window_att_config_builder_rf


def center_window_att_baseline_rf(
        win_size_list: Tuple[int, ...] = (5, 129),
        use_att_ctx_in_state: bool = True,
        use_weight_feedback: bool = True,
        blank_decoder_version: int = 3,
        blank_decoder_opts: Optional[dict] = None,
        use_current_frame_in_readout: bool = False,
        use_current_frame_in_readout_w_gate: bool = False,
        use_current_frame_in_readout_w_gate_v: int = 1,
        use_current_frame_in_readout_w_double_gate: bool = False,
        use_correct_dim_tags: bool = False,
        behavior_version: Optional[int] = None,
        bpe_vocab_size: int = 10025,
        use_sep_att_encoder: bool = False,
        use_sep_h_t_readout: bool = False,
):
  for win_size in win_size_list:
    alias, config_builder = get_center_window_att_config_builder_rf(
      win_size=win_size,
      use_att_ctx_in_state=use_att_ctx_in_state,
      blank_decoder_version=blank_decoder_version,
      use_joint_model=False,
      use_weight_feedback=use_weight_feedback,
      label_decoder_state="nb-lstm",
      blank_decoder_opts=blank_decoder_opts,
      use_current_frame_in_readout=use_current_frame_in_readout,
      use_current_frame_in_readout_w_gate=use_current_frame_in_readout_w_gate,
      use_current_frame_in_readout_w_gate_v=use_current_frame_in_readout_w_gate_v,
      use_current_frame_in_readout_w_double_gate=use_current_frame_in_readout_w_double_gate,
      use_correct_dim_tags=use_correct_dim_tags,
      behavior_version=behavior_version,
      bpe_vocab_size=bpe_vocab_size,
      use_sep_att_encoder=use_sep_att_encoder,
      use_sep_h_t_readout=use_sep_h_t_readout,
    )
    alias = f"{base_alias}/baseline_rf/{alias}"
    yield alias, config_builder
