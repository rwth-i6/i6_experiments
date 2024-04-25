import copy
from typing import Dict, List, Any, Optional, Tuple

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import LibrispeechConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att.baseline_v1.alias import alias as base_alias


def get_global_att_config_builder(use_weight_feedback: bool = True, decoder_version: Optional[int] = None):
  model_type = "librispeech_conformer_glob_att"
  variant_name = "glob.conformer.mohammad.5.6"
  variant_params = copy.deepcopy(models[model_type][variant_name])
  variant_params["network"]["use_weight_feedback"] = use_weight_feedback
  variant_params["network"]["decoder_version"] = decoder_version

  config_builder = LibrispeechConformerGlobalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


def global_att_baseline():
  alias = f"{base_alias}/baseline"
  yield alias, get_global_att_config_builder(use_weight_feedback=True)
