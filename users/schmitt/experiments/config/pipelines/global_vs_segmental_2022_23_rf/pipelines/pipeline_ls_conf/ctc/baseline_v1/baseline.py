from typing import Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.ctc.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.ctc.config_builder import get_ctc_config_builder_rf


def ctc_baseline_rf(
        label_type: str = "bpe10025",
        conformer_w_abs_pos_enc: bool = False,
        conformer_wo_rel_pos_enc: bool = False,
        conformer_wo_final_layer_norm_per_layer: bool = False,
        num_layers: int = 12
):
  alias, config_builder = get_ctc_config_builder_rf(
    label_type=label_type,
    conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
    conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
    conformer_wo_final_layer_norm_per_layer=conformer_wo_final_layer_norm_per_layer,
    num_layers=num_layers
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  yield alias, config_builder
