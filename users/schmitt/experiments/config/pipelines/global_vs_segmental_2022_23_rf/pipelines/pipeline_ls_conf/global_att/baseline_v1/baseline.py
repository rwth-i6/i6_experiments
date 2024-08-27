from typing import Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.config_builder import get_global_att_config_builder_rf


def global_att_baseline_rf(
        use_weight_feedback: bool = True,
        use_att_ctx_in_state: bool = True,
        decoder_state: str = "nb-lstm",
        label_type: str = "bpe10025",
        conformer_w_abs_pos_enc: bool = False,
        conformer_wo_rel_pos_enc: bool = False,
        disable_enc_self_att_until_epoch: Optional[int] = None,
):
  alias, config_builder = get_global_att_config_builder_rf(
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    decoder_state=decoder_state,
    label_type=label_type,
    conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
    conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
    disable_enc_self_att_until_epoch=disable_enc_self_att_until_epoch,
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  yield alias, config_builder
