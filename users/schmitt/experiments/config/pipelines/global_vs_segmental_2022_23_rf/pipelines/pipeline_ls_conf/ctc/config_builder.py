from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechCtcAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.ctc.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_CTC_ALIGNMENT,
  LibrispeechBPE1056_ALIGNMENT_SEP_MODEL,
  LibrispeechBPE5048_ALIGNMENT_SEP_MODEL,
  LibrispeechSP10240_ALIGNMENT_SEP_MODEL,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT


def get_ctc_config_builder_rf(
        label_type: str = "bpe10025",
        conformer_w_abs_pos_enc: bool = False,
        conformer_wo_rel_pos_enc: bool = False,
        conformer_wo_final_layer_norm_per_layer: bool = False,
        num_layers: int = 12
):
  if label_type == "bpe10025":
    dependencies = LibrispeechBPE10025_CTC_ALIGNMENT
  elif label_type == "bpe5048":
    dependencies = LibrispeechBPE5048_ALIGNMENT_SEP_MODEL
  elif label_type == "bpe1056":
    dependencies = LibrispeechBPE1056_ALIGNMENT_SEP_MODEL
  else:
    assert label_type == "sp10240"
    dependencies = LibrispeechSP10240_ALIGNMENT_SEP_MODEL

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

  config_builder = LibrispeechCtcAttConformerConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
    conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
    conformer_wo_final_layer_norm_per_layer=conformer_wo_final_layer_norm_per_layer,
    conformer_num_layers=num_layers,
    use_correct_dim_tags=True,
  )

  alias = (
    f"{label_type}/"
  )

  alias += f"{num_layers}-layer_"
  if (not conformer_w_abs_pos_enc) and (not conformer_wo_rel_pos_enc) and (not conformer_wo_final_layer_norm_per_layer):
    alias += "standard-conformer"
  else:
    alias += "conformer"
    if conformer_w_abs_pos_enc:
      alias += "-w-abs-pos"
    if conformer_wo_rel_pos_enc:
      alias += "-wo-rel-pos"
    if conformer_wo_final_layer_norm_per_layer:
      alias += "-wo-final-layer-norm-per-layer"

  return alias, config_builder
