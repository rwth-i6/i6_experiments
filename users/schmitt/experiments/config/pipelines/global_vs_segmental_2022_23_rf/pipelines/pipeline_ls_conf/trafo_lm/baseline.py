from typing import Optional, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm.config_builder import get_trafo_lm_config_builder_rf


def global_att_baseline_rf(
        label_type: str,
        num_layers: int = 12,
        model_dim: int = 512,
        pos_enc: Optional[str] = "rf.sinusoidal_positional_encoding",
        norm: str = "rf.LayerNorm",
        ff: str = "returnn.util.basic.NotSpecified",
        decoder_layer_opts: Optional[Dict] = None,
):
  alias, config_builder = get_trafo_lm_config_builder_rf(
    label_type=label_type,
    num_layers=num_layers,
    model_dim=model_dim,
    pos_enc=pos_enc,
    norm=norm,
    ff=ff,
    decoder_layer_opts=decoder_layer_opts,
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  yield alias, config_builder
