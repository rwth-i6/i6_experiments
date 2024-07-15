from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseLabelDecoder


class GlobalAttDecoder(BaseLabelDecoder):
  def __init__(self, eos_idx: int, **kwargs):
    super(GlobalAttDecoder, self).__init__(**kwargs)

    assert not self.use_current_frame_in_readout, "Not supported yet"

    self.eos_idx = eos_idx
    self.bos_idx = eos_idx

  def decoder_default_initial_state(
          self,
          *,
          batch_dims: Sequence[Dim],
          enc_spatial_dim: Dim,
          use_mini_att: bool = False,
  ) -> rf.State:
    """Default initial state"""
    state = rf.State(
      att=rf.zeros(list(batch_dims) + [self.att_num_heads * self.enc_out_dim]),
    )
    state.att.feature_dim_axis = len(state.att.dims) - 1

    if "lstm" in self.decoder_state:
      state.s = self.get_lstm().default_initial_state(batch_dims=batch_dims)
      if self.use_mini_att and "lstm":
        state.mini_att_lstm = self.mini_att_lstm.default_initial_state(batch_dims=batch_dims)

    if self.use_weight_feedback and not use_mini_att:
      state.accum_att_weights = rf.zeros(
        list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
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

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_ctx: rf.Tensor,
          inv_fertility: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          state: Optional[rf.State] = None,
          use_mini_att: bool = False,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    state_ = rf.State()

    prev_att = state.att
    prev_s_state = state.s if "lstm" in self.decoder_state else None

    # s, state_.s = self.s(rf.concat_features(input_embed, prev_att), state=state.s, spatial_dim=single_step_dim)
    s, s_state = self._update_state(input_embed, prev_att, prev_s_state)
    if "lstm" in self.decoder_state:
      state_.s = s_state

    if use_mini_att:
      if "lstm" in self.decoder_state:
        att_lstm, state_.mini_att_lstm = self.mini_att_lstm(
          input_embed, state=state.mini_att_lstm, spatial_dim=single_step_dim)
        pre_mini_att = att_lstm
      else:
        att_linear = self.mini_att_linear(input_embed)
        pre_mini_att = att_linear
      att = self.mini_att(pre_mini_att)
    else:
      if self.use_weight_feedback:
        weight_feedback = self.weight_feedback(state.accum_att_weights)
      else:
        weight_feedback = rf.zeros((self.enc_key_total_dim,))

      s_transformed = self.s_transformed(s)
      energy_in = enc_ctx + weight_feedback + s_transformed
      energy = self.energy(rf.tanh(energy_in))
      att_weights = rf.softmax(energy, axis=enc_spatial_dim)

      if self.use_weight_feedback:
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5

      att = self.get_att(att_weights, enc, enc_spatial_dim)

    state_.att = att

    return {"s": s, "att": att}, state_

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits


class GlobalAttEfficientDecoder(GlobalAttDecoder):
  def __init__(self, **kwargs):
    super(GlobalAttEfficientDecoder, self).__init__(**kwargs)

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
          input_embed: rf.Tensor,
          input_embed_spatial_dim: Dim,
          use_mini_att: bool = False,
  ) -> rf.Tensor:
    if use_mini_att:
      if "lstm" in self.decoder_state:
        att_lstm, _ = self.mini_att_lstm(
          input_embed,
          state=self.mini_att_lstm.default_initial_state(
            batch_dims=input_embed.remaining_dims([input_embed_spatial_dim, input_embed.feature_dim])),
          spatial_dim=input_embed_spatial_dim
        )
        pre_mini_att = att_lstm
      else:
        att_linear = self.mini_att_linear(input_embed)
        pre_mini_att = att_linear
      att = self.mini_att(pre_mini_att)
    else:
      s_transformed = self.s_transformed(s)

      weight_feedback = rf.zeros((self.enc_key_total_dim,))

      energy_in = enc_ctx + weight_feedback + s_transformed
      energy = self.energy(rf.tanh(energy_in))
      att_weights = rf.softmax(energy, axis=enc_spatial_dim)
      # we do not need use_mask because the softmax output is already padded with zeros
      att = self.get_att(att_weights, enc, enc_spatial_dim)

    return att

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor) -> Tensor:
    """logits for the decoder"""
    readout_in = self.readout_in(rf.concat_features(s, input_embed, att, allow_broadcast=True))
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits
