from typing import Optional, Dict, Any, Sequence, Tuple, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import BlankDecoder

from returnn.tensor import Dim
import returnn.frontend as rf


def viterbi_training(
        *,
        model: BlankDecoder,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
):
  align_input_embeddings = model.target_embed(align_targets)
  align_input_embeddings = rf.shift_right(
    align_input_embeddings, axis=align_targets_spatial_dim, pad_value=0.0)

  blank_loop_out, _ = model.loop_step(
    enc=enc_args["enc"],
    enc_spatial_dim=enc_spatial_dim,
    input_embed=align_input_embeddings,
    state=model.default_initial_state(batch_dims=batch_dims,),
    spatial_dim=align_targets_spatial_dim,
  )

  blank_logits = model.decode_logits(**blank_loop_out)
  blank_logits_packed, pack_dim = rf.pack_padded(
    blank_logits, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False)
  emit_ground_truth_packed, _ = rf.pack_padded(
    emit_ground_truth, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  # rf.log_sigmoid not implemented for torch backend
  emit_log_prob = rf.log(rf.sigmoid(blank_logits_packed))
  blank_log_prob = rf.log(rf.sigmoid(-blank_logits_packed))
  blank_logit_dim = blank_logits_packed.remaining_dims((pack_dim,))[0]
  emit_blank_log_prob, _ = rf.concat(
    (blank_log_prob, blank_logit_dim), (emit_log_prob, blank_logit_dim), out_dim=emit_blank_target_dim)
  blank_loss = rf.cross_entropy(
    target=emit_ground_truth_packed,
    estimated=emit_blank_log_prob,
    estimated_type="log-probs",
    axis=emit_blank_target_dim
  )
  blank_loss.mark_as_loss("emit_blank_ce", scale=1.0, use_normalized_loss=True)

  best = rf.reduce_argmax(emit_blank_log_prob, axis=emit_blank_target_dim)
  frame_error = best != emit_ground_truth_packed
  frame_error.mark_as_loss(name="emit_blank_fer", as_error=True)
