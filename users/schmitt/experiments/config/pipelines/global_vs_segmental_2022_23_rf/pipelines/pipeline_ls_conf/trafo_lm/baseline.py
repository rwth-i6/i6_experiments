from typing import Optional, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.pipelines.pipeline_ls_conf.trafo_lm.config_builder import get_trafo_lm_config_builder_rf


def global_att_baseline_rf(
        label_type: str = "bpe10025",
):
  alias, config_builder = get_trafo_lm_config_builder_rf(
    label_type=label_type,
  )
  alias = f"{base_alias}/baseline_rf/{alias}"
  yield alias, config_builder
