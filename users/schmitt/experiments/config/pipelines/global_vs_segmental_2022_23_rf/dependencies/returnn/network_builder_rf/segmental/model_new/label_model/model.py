from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseLabelDecoder


class SegmentalAttLabelDecoder(BaseLabelDecoder):
  def __init__(self, center_window_size: int, **kwargs):
    super(SegmentalAttLabelDecoder, self).__init__(**kwargs)

    self.center_window_size = center_window_size
    self.accum_att_weights_dim = Dim(name="accum_att_weights", dimension=center_window_size)

  def default_initial_state(
          self,
          *,
          batch_dims: Sequence[Dim],
          segment_starts_sparse_dim: Optional[Dim] = None,
          segment_lens_sparse_dim: Optional[Dim] = None,
  ) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s=self.get_lstm().default_initial_state(batch_dims=batch_dims),
      att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.enc_out_dim]),
      segment_starts=rf.zeros(batch_dims, sparse_dim=segment_starts_sparse_dim, dtype="int32"),
      segment_lens=rf.zeros(batch_dims, sparse_dim=segment_lens_sparse_dim, dtype="int32"),
    )
    state.att.feature_dim_axis = len(state.att.dims) - 1

    if self.use_weight_feedback:
      state.accum_att_weights = rf.zeros(
        list(batch_dims) + [self.accum_att_weights_dim, self.att_num_heads], feature_dim=self.att_num_heads
      )

    return state

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s": Tensor(
        "s", dims=batch_dims + [self.get_lstm().out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
      ),
      "att": Tensor(
        "att",
        dims=batch_dims + [self.att_num_heads * self.enc_out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1,
      ),
    }

  def _get_prev_accum_att_weights_scattered(
          self,
          prev_accum_att_weights: Tensor,
          segment_starts: Tensor,
          prev_segment_starts: Tensor,
          prev_segment_lens: Tensor,
  ) -> Tensor:

    overlap_len = rf.cast(prev_segment_starts + prev_segment_lens - segment_starts, "int32")
    overlap_len = rf.where(
      rf.logical_or(overlap_len < 0, overlap_len > prev_segment_lens),
      rf.convert_to_tensor(0),
      overlap_len
    )
    overlap_start = prev_segment_lens - overlap_len

    slice_dim = Dim(name="slice", dimension=overlap_len)
    gather_positions = rf.range_over_dim(slice_dim)
    gather_positions += overlap_start
    prev_accum_att_weights_overlap = rf.gather(
      prev_accum_att_weights, axis=self.accum_att_weights_dim, indices=gather_positions, clip_to_valid=True
    )
    overlap_range = rf.range_over_dim(slice_dim)

    prev_accum_att_weights_scattered = rf.scatter(
      prev_accum_att_weights_overlap,
      out_dim=self.accum_att_weights_dim,
      indices=overlap_range,
      indices_dim=slice_dim,
    )

    return prev_accum_att_weights_scattered

  def _get_accum_att_weights(
          self,
          att_t_dim: Dim,
          enc_spatial_dim: Dim,
          inv_fertility: Tensor,
          att_weights: Tensor,
          prev_accum_att_weights_scattered: Tensor,
          gather_positions: Tensor,
  ) -> Tensor:
    att_weights_range = rf.range_over_dim(att_t_dim)
    att_weights_scattered = rf.scatter(
      att_weights,
      out_dim=self.accum_att_weights_dim,
      indices=att_weights_range,
      indices_dim=att_t_dim,
    )

    inv_fertility_sliced = rf.gather(inv_fertility, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
    inv_fertility_scattered = rf.scatter(
      inv_fertility_sliced,
      out_dim=self.accum_att_weights_dim,
      indices=att_weights_range,
      indices_dim=att_t_dim,
    )

    accum_att_weights = prev_accum_att_weights_scattered + att_weights_scattered * inv_fertility_scattered * 0.5

    return accum_att_weights

  def _get_weight_feedback(
          self,
          prev_accum_att_weights_scattered: Tensor,
          att_t_dim: Dim,
  ) -> Tensor:
    gather_positions = rf.range_over_dim(att_t_dim)
    prev_accum_att_weights_sliced = rf.gather(
      prev_accum_att_weights_scattered,
      axis=self.accum_att_weights_dim,
      indices=gather_positions,
      clip_to_valid=True
    )

    return self.weight_feedback(prev_accum_att_weights_sliced)

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          segment_starts: rf.Tensor,
          segment_lens: rf.Tensor,
          state: rf.State,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    state_ = rf.State()

    # during search, these need to be the values from the previous "emit" step (not necessarily the previous time step)
    prev_att = state.att
    prev_s_state = state.s
    prev_segment_starts = state.segment_starts
    prev_segment_lens = state.segment_lens

    s, state_.s = self._update_state(input_embed, prev_att, prev_s_state)
    s_transformed = self.s_transformed(s)

    slice_dim = Dim(name="slice", dimension=segment_lens)
    gather_positions = rf.range_over_dim(slice_dim)
    gather_positions += segment_starts

    enc_ctx_sliced = rf.gather(enc_ctx, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
    enc_sliced = rf.gather(enc, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)

    if self.use_weight_feedback:
      prev_accum_att_weights = state.accum_att_weights
      prev_accum_att_weights_scattered = self._get_prev_accum_att_weights_scattered(
        prev_accum_att_weights=prev_accum_att_weights,
        segment_starts=segment_starts,
        prev_segment_starts=prev_segment_starts,
        prev_segment_lens=prev_segment_lens,
      )
      weight_feedback = self._get_weight_feedback(
        prev_accum_att_weights_scattered=prev_accum_att_weights_scattered,
        att_t_dim=slice_dim,
      )
    else:
      weight_feedback = rf.zeros((self.enc_key_total_dim,))

    energy_in = enc_ctx_sliced + weight_feedback + s_transformed
    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=slice_dim)
    # we do not need use_mask because the softmax output is already padded with zeros
    att0 = rf.dot(att_weights, enc_sliced, reduce=slice_dim, use_mask=False)
    att0.feature_dim = self.enc_out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.enc_out_dim))
    state_.att = att

    if self.use_weight_feedback:
      accum_att_weights = self._get_accum_att_weights(
        att_t_dim=slice_dim,
        enc_spatial_dim=enc_spatial_dim,
        inv_fertility=inv_fertility,
        att_weights=att_weights,
        prev_accum_att_weights_scattered=prev_accum_att_weights_scattered,
        gather_positions=gather_positions,
      )
      accum_att_weights.feature_dim = self.att_num_heads
      state_.accum_att_weights = accum_att_weights

    state_.segment_starts = segment_starts
    state_.segment_lens = segment_lens

    return {"s": s, "att": att}, state_

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits


class SegmentalAttEfficientLabelDecoder(SegmentalAttLabelDecoder):
  def __init__(self, **kwargs):
    super(SegmentalAttEfficientLabelDecoder, self).__init__(**kwargs)

    assert not self.use_att_ctx_in_state and not self.use_weight_feedback, (
      "Cannot have alignment dependency for efficient implementation!"
    )

  def __call__(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          enc_spatial_dim: Dim,
          s: rf.Tensor,
          segment_starts: rf.Tensor,
          segment_lens: rf.Tensor,
  ) -> rf.Tensor:
    s_transformed = self.s_transformed(s)

    slice_dim = Dim(name="slice", dimension=segment_lens)
    gather_positions = rf.range_over_dim(slice_dim)
    gather_positions += segment_starts

    # need to move size tensor to GPU since otherwise there is an error in some merge_dims call inside rf.gather
    # because two tensors have different devices
    # TODO: fix properly in the gather implementation
    enc_spatial_dim.dyn_size_ext = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, gather_positions.device)

    enc_ctx_sliced = rf.gather(enc_ctx, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)
    enc_sliced = rf.gather(enc, axis=enc_spatial_dim, indices=gather_positions, clip_to_valid=True)

    # move size tensor back to CPU
    enc_spatial_dim.dyn_size_ext = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, "cpu")

    weight_feedback = rf.zeros((self.enc_key_total_dim,))

    energy_in = enc_ctx_sliced + weight_feedback + s_transformed

    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=slice_dim)
    # we do not need use_mask because the softmax output is already padded with zeros
    att0 = rf.dot(att_weights, enc_sliced, reduce=slice_dim, use_mask=False)
    att0.feature_dim = self.enc_out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.enc_out_dim))

    return att

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att, allow_broadcast=True))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits
