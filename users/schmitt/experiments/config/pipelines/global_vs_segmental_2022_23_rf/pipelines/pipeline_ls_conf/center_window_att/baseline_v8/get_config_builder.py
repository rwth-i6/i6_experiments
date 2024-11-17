from typing import Tuple, Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.baseline_v8.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.center_window_att.config_builder import get_center_window_att_config_builder_rf


def center_window_att_baseline_rf(
        win_size: int,
        window_step_size: int,
        use_att_ctx_in_state: bool = True,
        use_weight_feedback: bool = True,
        bpe_vocab_size: int = 10025,
):
  alias, config_builder = get_center_window_att_config_builder_rf(
    win_size=win_size,
    use_att_ctx_in_state=use_att_ctx_in_state,
    blank_decoder_version=None,
    use_joint_model=True,
    use_weight_feedback=use_weight_feedback,
    label_decoder_state="joint-lstm",
    use_correct_dim_tags=True,
    behavior_version=21,
    bpe_vocab_size=bpe_vocab_size,
    window_step_size=window_step_size,
    use_vertical_transitions=True,
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  return alias, config_builder
