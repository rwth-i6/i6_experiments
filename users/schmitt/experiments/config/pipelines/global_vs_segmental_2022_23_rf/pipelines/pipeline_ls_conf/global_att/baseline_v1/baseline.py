import copy
from typing import Dict, List, Any, Optional, Tuple

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import GlobalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att.baseline_v1.alias import alias as base_alias


def get_global_att_config_builder_rf(use_weight_feedback: bool = True, decoder_version: Optional[int] = None):
  model_type = "librispeech_conformer_glob_att"
  variant_name = "glob.conformer.mohammad.5.6"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["use_weight_feedback"] = use_weight_feedback
  variant_params["network"]["decoder_version"] = decoder_version

  config_builder = GlobalAttConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
  )

  return config_builder


def global_att_baseline_rf():
  alias = f"{base_alias}/baseline_rf"
  yield alias, get_global_att_config_builder_rf(use_weight_feedback=True)
