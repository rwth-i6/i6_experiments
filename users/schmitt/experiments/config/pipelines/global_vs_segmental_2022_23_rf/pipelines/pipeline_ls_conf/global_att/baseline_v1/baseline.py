from typing import Optional, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.config_builder import get_global_att_config_builder_rf


def global_att_baseline_rf(
        use_weight_feedback: bool = True,
        use_att_ctx_in_state: bool = True,
        decoder_state: str = "nb-lstm",
        label_type: str = "bpe10025",
        conformer_w_abs_pos_enc: bool = False,
        conformer_wo_rel_pos_enc: bool = False,
        conformer_wo_final_layer_norm_per_layer: bool = False,
        conformer_num_layers: int = 12,
        conformer_wo_convolution: bool = False,
        conformer_out_dim: int = 512,
        enc_ctx_layer: Optional["str"] = None,
        conformer_conv_w_zero_padding: bool = False,
        use_feed_forward_encoder: bool = False,
        hard_att_opts: Optional[Dict] = None,
        conv_frontend_w_zero_padding: bool = False,
        replace_att_by_h_s: bool = False,
):
  alias, config_builder = get_global_att_config_builder_rf(
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    decoder_state=decoder_state,
    label_type=label_type,
    conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
    conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
    conformer_wo_final_layer_norm_per_layer=conformer_wo_final_layer_norm_per_layer,
    conformer_num_layers=conformer_num_layers,
    conformer_wo_convolution=conformer_wo_convolution,
    conformer_out_dim=conformer_out_dim,
    enc_ctx_layer=enc_ctx_layer,
    conformer_conv_w_zero_padding=conformer_conv_w_zero_padding,
    use_feed_forward_encoder=use_feed_forward_encoder,
    hard_att_opts=hard_att_opts,
    conv_frontend_w_zero_padding=conv_frontend_w_zero_padding,
    replace_att_by_h_s=replace_att_by_h_s,
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  yield alias, config_builder
