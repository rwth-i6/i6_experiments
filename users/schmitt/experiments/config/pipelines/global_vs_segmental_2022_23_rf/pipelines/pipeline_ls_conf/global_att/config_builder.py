from typing import Optional

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.config_builder_rf.base import LibrispeechGlobalAttConformerConfigBuilderRF
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.global_.model import from_scratch_model_def, _returnn_v2_get_model
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.labels.v2.librispeech.label_singletons import (
  LibrispeechBPE10025_LABELS,
  LibrispeechSP10240_LABELS,
  LibrispeechBPE1056_LABELS,
  LibrispeechBPE5048_LABELS,
  LIBRISPEECH_CORPUS
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.general.returnn.exes import RETURNN_EXE_NEW, RETURNN_CURRENT_ROOT


def get_global_att_config_builder_rf(
        use_weight_feedback: bool = True,
        use_att_ctx_in_state: bool = True,
        decoder_state: str = "nb-lstm",
        num_label_decoder_layers: int = 1,
        label_type: str = "bpe10025",
        conformer_w_abs_pos_enc: bool = False,
        conformer_wo_rel_pos_enc: bool = False,
        conformer_wo_final_layer_norm_per_layer: bool = False,
        conformer_num_layers: int = 12,
        conformer_wo_convolution: bool = False,
        conformer_out_dim: int = 512,
        enc_ctx_layer: Optional[str] = None,
        conformer_conv_w_zero_padding: bool = False,
        use_feed_forward_encoder: bool = False,
        hard_att_opts: Optional[dict] = None,
        conv_frontend_w_zero_padding: bool = False,
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

  config_builder = LibrispeechGlobalAttConformerConfigBuilderRF(
    variant_params=variant_params,
    model_def=from_scratch_model_def,
    get_model_func=_returnn_v2_get_model,
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    label_decoder_state=decoder_state,
    num_label_decoder_layers=num_label_decoder_layers,
    conformer_w_abs_pos_enc=conformer_w_abs_pos_enc,
    conformer_wo_rel_pos_enc=conformer_wo_rel_pos_enc,
    conformer_wo_final_layer_norm_per_layer=conformer_wo_final_layer_norm_per_layer,
    conformer_num_layers=conformer_num_layers,
    conformer_wo_convolution=conformer_wo_convolution,
    conformer_out_dim=conformer_out_dim,
    enc_ctx_layer=enc_ctx_layer,
    conformer_conv_w_zero_padding=conformer_conv_w_zero_padding,
    use_feed_forward_encoder=use_feed_forward_encoder,
    hard_att_opts=hard_att_opts,
    conv_frontend_w_zero_padding=conv_frontend_w_zero_padding
  )

  alias = (
    f"{label_type}/"
    f"{'w' if use_weight_feedback else 'wo'}-weight-feedback/"
    f"{'w' if use_att_ctx_in_state else 'wo'}-att-ctx-in-state/"
    f"{decoder_state}{'_att_keys_from_' + enc_ctx_layer if enc_ctx_layer is not None else ''}/"
  )

  alias += f"{conformer_num_layers}-layer_{conformer_out_dim}-dim_"
  if use_feed_forward_encoder:
    alias += "ff-encoder"
  else:
    if (
            not conformer_w_abs_pos_enc) and (
            not conformer_wo_rel_pos_enc) and (
            not conformer_wo_final_layer_norm_per_layer) and (
            not conformer_conv_w_zero_padding) and (
            not conformer_wo_convolution) and (
            not conv_frontend_w_zero_padding
    ):
      alias += "standard-conformer"
    else:
      alias += "conformer"
      if conformer_w_abs_pos_enc:
        alias += "-w-abs-pos"
      if conformer_wo_rel_pos_enc:
        alias += "-wo-rel-pos"
      if conformer_wo_final_layer_norm_per_layer:
        alias += "-wo-final-layer-norm-per-layer"
      if conformer_wo_convolution:
        alias += "-wo-convolution"
      if conformer_conv_w_zero_padding:
        alias += "-conv-w-zero-padding"
      if conv_frontend_w_zero_padding:
        alias += "-conv-frontend-w-zero-padding"

  return alias, config_builder
