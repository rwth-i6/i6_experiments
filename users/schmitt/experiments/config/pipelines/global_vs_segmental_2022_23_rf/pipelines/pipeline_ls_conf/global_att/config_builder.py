from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import GlobalAttConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_LABELS,
LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT


def get_global_att_config_builder_rf(
        use_weight_feedback: bool = True,
        use_att_ctx_in_state: bool = True,
):
  variant_params = {
    "dependencies": LibrispeechBPE10025_LABELS,
    "dataset": {
      "feature_type": "raw",
      "corpus": LIBRISPEECH_CORPUS
    },
    "config": {
      "train_seq_ordering": "laplace:.1000"
    },
    "network": {"length_scale": 1.0},
    "returnn_python_exe": RETURNN_EXE_NEW,
    "returnn_root": RETURNN_CURRENT_ROOT
  }

  config_builder = GlobalAttConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
  )

  alias = (
    f"{'w' if use_weight_feedback else 'wo'}-weight-feedback/"
    f"{'w' if use_att_ctx_in_state else 'wo'}-att-ctx-in-state"
  )

  return alias, config_builder
