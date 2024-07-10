from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v4.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.config_builder import get_center_window_att_config_builder_rf


def center_window_att_baseline_rf(
        win_size_list: Tuple[int, ...] = (5, 129),
        use_att_ctx_in_state: bool = True,
        label_decoder_state: str = "nb-lstm",
        use_weight_feedback: bool = True,
        bpe_vocab_size: int = 10025,
        separate_blank_from_softmax: bool = False,
):
  for win_size in win_size_list:
    alias, config_builder = get_center_window_att_config_builder_rf(
      win_size=win_size,
      use_att_ctx_in_state=use_att_ctx_in_state,
      blank_decoder_version=None,
      use_joint_model=True,
      label_decoder_state=label_decoder_state,
      use_weight_feedback=use_weight_feedback,
      bpe_vocab_size=bpe_vocab_size,
      separate_blank_from_softmax=separate_blank_from_softmax,
    )
    alias = f"{base_alias}/baseline_rf/{alias}"
    yield alias, config_builder
