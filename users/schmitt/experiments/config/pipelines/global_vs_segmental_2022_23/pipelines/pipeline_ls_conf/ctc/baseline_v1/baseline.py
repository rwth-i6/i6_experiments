import copy

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.ctc.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.ctc import LibrispeechConformerCtcConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models


def _get_ctc_config_builder():
  model_type = "librispeech_conformer_ctc"
  variant_name = "ctc.conformer.mohammad.5.6"
  variant_params = copy.deepcopy(models[model_type][variant_name])

  config_builder = LibrispeechConformerCtcConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


def ctc_baseline():
  alias = f"{base_alias}/baseline"
  yield alias, _get_ctc_config_builder()
