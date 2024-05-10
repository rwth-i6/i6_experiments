import copy
from typing import Dict, List, Any, Optional, Tuple

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.config_builder.global_ import Tedlium2ConformerGlobalAttentionConfigBuilder
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.model_variants.model_variants_ls_conf import models
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.pipelines.pipeline_ls_conf.global_att.baseline_v1.alias import alias as base_alias
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.corpora.tedlium2 import TedLium2Corpora
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.tedlium2.label_singletons import (
  TEDLIUM2BPE1058_LABELS
)

TEDLIUM_CORPUS = TedLium2Corpora()


def get_global_att_config_builder():
  variant_params = {
    "dependencies": TEDLIUM2BPE1058_LABELS,
    "dataset": {
      "feature_type": "raw",
      "corpus": TEDLIUM_CORPUS
    },
    "network": {},
    "config": {
      "train_seq_ordering": "laplace:.1000"
    },
    "returnn_python_exe": RETURNN_EXE_NEW,
    "returnn_root": RETURNN_CURRENT_ROOT
  }

  config_builder = Tedlium2ConformerGlobalAttentionConfigBuilder(
    dependencies=variant_params["dependencies"],
    variant_params=variant_params,
  )

  return config_builder


def global_att_baseline():
  alias = f"{base_alias}"
  yield alias, get_global_att_config_builder()
