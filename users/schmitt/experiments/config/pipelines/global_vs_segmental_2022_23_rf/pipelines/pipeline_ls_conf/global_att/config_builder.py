from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechGlobalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_LABELS,
  LibrispeechSP10240_LABELS,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT


def get_global_att_config_builder_rf(
        use_weight_feedback: bool = True,
        use_att_ctx_in_state: bool = True,
        decoder_state: str = "nb-lstm",
        num_label_decoder_layers: int = 1,
        label_type: str = "bpe10025",
):
  if label_type == "bpe10025":
    dependencies = LibrispeechBPE10025_LABELS
  else:
    assert label_type == "sp10240"
    dependencies = LibrispeechSP10240_LABELS

  variant_params = {
    "dependencies": dependencies,
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

  config_builder = LibrispeechGlobalAttConformerConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    label_decoder_state=decoder_state,
    num_label_decoder_layers=num_label_decoder_layers,
  )

  alias = (
    f"{'w' if use_weight_feedback else 'wo'}-weight-feedback/"
    f"{'w' if use_att_ctx_in_state else 'wo'}-att-ctx-in-state/"
    f"{decoder_state}"
  )

  return alias, config_builder
