from typing import Optional, Dict, Any, Sequence, Tuple, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder_rf.base import BaseModel

from returnn.returnn.tensor import Tensor, Dim, single_step_dim
import returnn.returnn.frontend as rf
from returnn.returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample


class SegmentalAttentionModel(BaseModel):
  def __init__(
          self,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          center_window_size: int,
          **kwargs
  ):
    super(SegmentalAttentionModel, self).__init__(**kwargs)

    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)
    self.accum_att_weights_dim = Dim(name="accum_att_weights", dimension=center_window_size)

    self.target_embed_length_model = rf.Linear(self.target_dim, self.length_model_embed_dim, with_bias=False)
    self.s_length_model = rf.LSTM(self.encoder.out_dim + self.length_model_embed_dim, self.length_model_state_dim)
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  # TODO: the dimensions of the initial accumulated attention weights need to be changed
  def decoder_default_initial_state(self, *, batch_dims: Sequence[Dim], enc_spatial_dim: Dim) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s=self.s.default_initial_state(batch_dims=batch_dims),
      att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.encoder.out_dim]),
      accum_att_weights=rf.zeros(
        list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
      ),
    )
    state.att.feature_dim_axis = len(state.att.dims) - 1
    return state

  def _get_weight_feedback(
          self,
          prev_accum_att_weights: rf.Tensor,
          segment_starts: rf.Tensor,
          prev_segment_starts: rf.Tensor,
          prev_segment_lens: rf.Tensor,
  ) -> rf.Tensor:

    overlap_len = prev_segment_starts + prev_segment_lens - segment_starts
    overlap_len = rf.cond(overlap_len < 0, lambda: 0, lambda: overlap_len)
    overlap_start = prev_segment_lens - overlap_len

    prev_accum_att_weights_overlap, overlap_dim = rf.slice(prev_accum_att_weights, axis=self.accum_att_weights_dim, start=overlap_start, size=overlap_len)
    overlap_range = rf.range_over_dim()

  def label_sync_loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          segment_starts: rf.Tensor,
          segment_lens: rf.Tensor,
          state: Optional[rf.State] = None,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    state_ = rf.State()

    prev_att = state.att

    s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)

    weight_feedback = self._get_weight_feedback()
    s_transformed = self.s_transformed(s)

    enc_ctx_sliced, att_t_dim = rf.slice(enc_ctx, axis=enc_spatial_dim, start=segment_starts, size=segment_lens)
    enc_sliced, _ = rf.slice(enc, axis=enc_spatial_dim, start=segment_starts, size=segment_lens, out_dim=att_t_dim)

    energy_in = enc_ctx_sliced + weight_feedback + s_transformed
    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=att_t_dim)
    state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5  # TODO: this needs to be changed
    att0 = rf.dot(att_weights, enc_sliced, reduce=att_t_dim, use_mask=True)
    att0.feature_dim = self.encoder.out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
    state_.att = att

    return {"s": s, "att": att}, state_
