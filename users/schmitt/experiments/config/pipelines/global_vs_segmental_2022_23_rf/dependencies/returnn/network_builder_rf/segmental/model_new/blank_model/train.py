from typing import Optional, Dict, Any, Sequence, Tuple, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental.model_new.blank_model.model import (
  BlankDecoderV1,
  BlankDecoderV3,
  BlankDecoderV4,
  BlankDecoderV5,
  BlankDecoderV6,
  BlankDecoderV7,
  BlankDecoderV8,
  BlankDecoderV9,
)
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils

from returnn.tensor import Dim
import returnn.frontend as rf


def get_packed_logits_and_emit_ground_truth(
        *,
        blank_logits: rf.Tensor,
        align_targets_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_prob_dim: Dim,
        batch_dims: List[Dim],
        beam_dim: Optional[Dim] = None,
):
  if beam_dim is not None:
    batch_dims = blank_logits.remaining_dims([beam_dim, align_targets_spatial_dim, emit_prob_dim])

  blank_logits_packed, pack_dim = rf.pack_padded(
    blank_logits, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False)
  emit_ground_truth_packed, _ = rf.pack_padded(
    emit_ground_truth, dims=batch_dims + [align_targets_spatial_dim], enforce_sorted=False, out_dim=pack_dim
  )

  return blank_logits_packed, pack_dim, emit_ground_truth_packed


def calc_loss(
        *,
        blank_logits_packed: rf.Tensor,
        emit_ground_truth_packed: rf.Tensor,
        emit_blank_target_dim: Dim,
        blank_logit_dim: Dim,
) -> Tuple[rf.Tensor, rf.Tensor]:
  # rf.log_sigmoid not implemented for torch backend
  emit_log_prob = rf.log(rf.sigmoid(blank_logits_packed))
  blank_log_prob = rf.log(rf.sigmoid(-blank_logits_packed))
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

  return emit_log_prob, blank_log_prob


def viterbi_training(
        *,
        model: BlankDecoderV1,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        align_targets: rf.Tensor,
        align_targets_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
        beam_dim: Optional[Dim] = None,
) -> Tuple[rf.Tensor, rf.Tensor]:
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

  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=model.decode_logits(**blank_loop_out),
    align_targets_spatial_dim=align_targets_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims,
    beam_dim=beam_dim
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )


def viterbi_training_v3(
        *,
        model: BlankDecoderV3,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        label_states_unmasked: rf.Tensor,
        label_states_unmasked_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  blank_loop_out, _ = model.loop_step(
    enc=enc_args["enc"],
    enc_spatial_dim=enc_spatial_dim,
    label_model_state=label_states_unmasked,
    state=model.default_initial_state(batch_dims=batch_dims,),
    spatial_dim=label_states_unmasked_spatial_dim,
  )

  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=model.decode_logits(**blank_loop_out),
    align_targets_spatial_dim=label_states_unmasked_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )


def viterbi_training_v4(
        *,
        model: BlankDecoderV4,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        label_states_unmasked: rf.Tensor,
        label_states_unmasked_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  # using dim.declare_same_as() leads to an error after an epoch is finished
  # (UnicodeDecodeError: 'ascii' codec can't decode byte 0xe2 in position 0: ordinal not in range(128))
  # therefore, we use the following workaround
  enc = enc_args["enc"]  # type: rf.Tensor
  enc = utils.copy_tensor_replace_dim_tag(enc, enc_spatial_dim, label_states_unmasked_spatial_dim)

  blank_logits = model.decode_logits(enc=enc, label_model_states_unmasked=label_states_unmasked)
  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=blank_logits,
    align_targets_spatial_dim=label_states_unmasked_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )


def viterbi_training_v5(
        *,
        model: BlankDecoderV5,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        label_states_unmasked: rf.Tensor,
        label_states_unmasked_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  # using dim.declare_same_as() leads to an error after an epoch is finished (see viterbi_training_v4)
  enc = enc_args["enc"]  # type: rf.Tensor
  enc = utils.copy_tensor_replace_dim_tag(enc, enc_spatial_dim, label_states_unmasked_spatial_dim)

  blank_logits = model.emit_prob(rf.concat_features(enc, label_states_unmasked))
  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=blank_logits,
    align_targets_spatial_dim=label_states_unmasked_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )


def viterbi_training_v6(
        *,
        model: BlankDecoderV6,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        label_states_unmasked: rf.Tensor,
        label_states_unmasked_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  # using dim.declare_same_as() leads to an error after an epoch is finished (see viterbi_training_v4)
  enc = enc_args["enc"]  # type: rf.Tensor
  enc = utils.copy_tensor_replace_dim_tag(enc, enc_spatial_dim, label_states_unmasked_spatial_dim)

  s, _ = model.s(
    enc,
    state=model.s.default_initial_state(batch_dims=batch_dims,),
    spatial_dim=label_states_unmasked_spatial_dim
  )
  blank_logits = model.emit_prob(rf.concat_features(s, label_states_unmasked))
  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=blank_logits,
    align_targets_spatial_dim=label_states_unmasked_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )

def viterbi_training_v7(
        *,
        model: BlankDecoderV7,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        label_states_unmasked: rf.Tensor,
        label_states_unmasked_spatial_dim: Dim,
        emit_positions_unmasked: rf.Tensor,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  # using dim.declare_same_as() leads to an error after an epoch is finished (see viterbi_training_v4)
  enc = enc_args["enc"]  # type: rf.Tensor
  enc = utils.copy_tensor_replace_dim_tag(enc, enc_spatial_dim, label_states_unmasked_spatial_dim)

  prev_emit_distances = rf.range_over_dim(
    label_states_unmasked_spatial_dim, dtype="int32") - emit_positions_unmasked
  prev_emit_distances -= 1  # 0-based to index embedding
  prev_emit_distances = rf.clip_by_value(prev_emit_distances, 0, model.distance_dim.dimension - 1)
  prev_emit_distances.sparse_dim = model.distance_dim

  blank_logits = model.decode_logits(
    enc=enc,
    label_model_states_unmasked=label_states_unmasked,
    prev_emit_distances=prev_emit_distances
  )
  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=blank_logits,
    align_targets_spatial_dim=label_states_unmasked_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )


def viterbi_training_v8(
        *,
        model: BlankDecoderV8,
        enc_args: Dict,
        enc_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  emit_ground_truth_spatial_dim = emit_ground_truth.remaining_dims(batch_dims)[0]

  # using dim.declare_same_as() leads to an error after an epoch is finished (see viterbi_training_v4)
  enc = enc_args["enc"]  # type: rf.Tensor
  enc = utils.copy_tensor_replace_dim_tag(enc, enc_spatial_dim, emit_ground_truth_spatial_dim)

  blank_logits = model.decode_logits(enc=enc)
  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=blank_logits,
    align_targets_spatial_dim=emit_ground_truth_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )


def viterbi_training_v9(
        *,
        model: BlankDecoderV9,
        energy_in: rf.Tensor,
        energy_in_spatial_dim: Dim,
        non_blank_mask: rf.Tensor,
        enc_spatial_dim: Dim,
        emit_ground_truth: rf.Tensor,
        emit_blank_target_dim: Dim,
        batch_dims: List[Dim],
) -> Tuple[rf.Tensor, rf.Tensor]:
  blank_logits = model.decode_logits(energy_in=energy_in)

  emit_ground_truth_spatial_dim = emit_ground_truth.remaining_dims(batch_dims)[0]
  blank_logits_unmasked = utils.get_unmasked(
    input=blank_logits,
    input_spatial_dim=energy_in_spatial_dim,
    mask=non_blank_mask,
    mask_spatial_dim=emit_ground_truth_spatial_dim
  )

  indices = rf.ones(dims=batch_dims + [emit_ground_truth_spatial_dim], dtype="int32")
  indices = utils.cumsum(indices, dim=emit_ground_truth_spatial_dim) - 1
  # singleton_dim = rf.Dim(name="singleton", dimension=1)
  # indices = rf.expand_dim(indices, singleton_dim) - 1  # minus 1 because zero-based
  indices.sparse_dim = enc_spatial_dim

  blank_logits = rf.gather(
    blank_logits_unmasked,
    indices=indices,
    axis=enc_spatial_dim,
    clip_to_valid=True,
  )

  blank_logits_packed, pack_dim, emit_ground_truth_packed = get_packed_logits_and_emit_ground_truth(
    blank_logits=blank_logits,
    align_targets_spatial_dim=emit_ground_truth_spatial_dim,
    emit_ground_truth=emit_ground_truth,
    emit_prob_dim=model.emit_prob.out_dim,
    batch_dims=batch_dims
  )

  return calc_loss(
    blank_logits_packed=blank_logits_packed,
    emit_ground_truth_packed=emit_ground_truth_packed,
    emit_blank_target_dim=emit_blank_target_dim,
    blank_logit_dim=model.emit_prob.out_dim
  )
