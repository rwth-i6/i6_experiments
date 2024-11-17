from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.baseline_v2.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.global_att.config_builder import get_global_att_config_builder_rf


def global_att_baseline_rf(
        use_weight_feedback: bool = True,
        use_att_ctx_in_state: bool = True,
        label_type: str = "sp10240",
):
  alias, config_builder = get_global_att_config_builder_rf(
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    decoder_state="trafo",
    num_label_decoder_layers=6,
    label_type=label_type,
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  yield alias, config_builder
