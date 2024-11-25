from typing import Optional, Dict

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.lm import LibrispeechLstmLmConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.lm.trafo.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_LABELS,
  LibrispeechSP10240_LABELS,
  LibrispeechBPE1056_LABELS,
  LibrispeechBPE5048_LABELS,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT


def get_trafo_lm_config_builder_rf(
        label_type: str,
        num_layers: int,
        model_dim: int,
        pos_enc: Optional[str],
        ff: str,
        norm: str,
        decoder_layer_opts: Optional[Dict],
):
  if label_type == "bpe10025":
    dependencies = LibrispeechBPE10025_LABELS
  elif label_type == "bpe5048":
    dependencies = LibrispeechBPE5048_LABELS
  elif label_type == "bpe1056":
    dependencies = LibrispeechBPE1056_LABELS
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

  config_builder = LibrispeechLstmLmConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    num_layers=num_layers,
    model_dim=model_dim,
    pos_enc=pos_enc,
    ff=ff,
    norm=norm,
    decoder_layer_opts=decoder_layer_opts,
  )

  alias = (
    f"{label_type}_{num_layers}-layers_{model_dim}-dim_pos-enc-{pos_enc}_ff-{ff}_norm-{norm}"
    # f"{'' if decoder_layer_opts is None else '_'.join(f'{k}-{v}' for k, v in decoder_layer_opts.items())}"
  )

  return alias, config_builder
