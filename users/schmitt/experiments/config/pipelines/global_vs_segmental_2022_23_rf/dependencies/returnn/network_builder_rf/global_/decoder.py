from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseLabelDecoder


class GlobalAttDecoder(BaseLabelDecoder):
  def __init__(self, eos_idx: int, **kwargs):
    super(GlobalAttDecoder, self).__init__(**kwargs)

    self.eos_idx = eos_idx
    self.bos_idx = eos_idx

  def decoder_default_initial_state(
          self,
          *,
          batch_dims: Sequence[Dim],
          enc_spatial_dim: Dim,
          use_mini_att: bool = False,
          use_zero_att: bool = False,
  ) -> rf.State:
    """Default initial state"""
    state = rf.State()

    if self.trafo_att and not use_mini_att and not use_zero_att:
      state.trafo_att = self.trafo_att.default_initial_state(batch_dims=batch_dims)
      # att_dim = self.trafo_att.model_dim
    # else:
      # att_dim = self.att_num_heads * self.enc_out_dim

    state.att = rf.zeros(list(batch_dims) + [self.att_dim])
    state.att.feature_dim_axis = len(state.att.dims) - 1

    if "lstm" in self.decoder_state:
      state.s = self.get_lstm().default_initial_state(batch_dims=batch_dims)
      if self.use_mini_att and "lstm":
        state.mini_att_lstm = self.mini_att_lstm.default_initial_state(batch_dims=batch_dims)

    if self.use_weight_feedback and not use_mini_att and not use_zero_att:
      state.accum_att_weights = rf.zeros(
        list(batch_dims) + [enc_spatial_dim, self.att_num_heads], feature_dim=self.att_num_heads
      )

    return state

  def loop_step_output_templates(
          self,
          batch_dims: List[Dim],
  ) -> Dict[str, Tensor]:
    """loop step out"""
    output_templates = {
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
    return output_templates

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
          use_zero_att: bool = False,
          hard_att_opts: Optional[Dict] = None,
          mask_att_opts: Optional[Dict] = None,
          detach_att: bool = False,
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.decoder_default_initial_state(batch_dims=batch_dims, enc_spatial_dim=enc_spatial_dim)
    state_ = rf.State()

    output_dict = {}

    prev_att = state.att
    if detach_att:
      prev_att = rf.stop_gradient(prev_att)
    prev_s_state = state.s if "lstm" in self.decoder_state else None

    input_embed = rf.dropout(input_embed, drop_prob=self.target_embed_dropout, axis=None)
    s, s_state = self._update_state(input_embed, prev_att, prev_s_state)
    output_dict["s"] = s
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
    elif use_zero_att:
      att = rf.zeros_like(prev_att)
    else:
      if self.trafo_att:
        att, state_.trafo_att = self.trafo_att(
          input_embed,
          state=state.trafo_att,
          spatial_dim=single_step_dim,
          encoder=None if self.use_trafo_att_wo_cross_att else self.trafo_att.transform_encoder(enc, axis=enc_spatial_dim)
        )
      else:
        if self.use_weight_feedback:
          weight_feedback = self.weight_feedback(state.accum_att_weights)
        else:
          weight_feedback = rf.zeros((self.enc_key_total_dim,))

        if hard_att_opts is None or rf.get_run_ctx().epoch > hard_att_opts["until_epoch"]:
          s_transformed = self.s_transformed(s)
          energy_in = enc_ctx + weight_feedback + s_transformed

          energy = self.energy(rf.tanh(energy_in))

          # set energies to -inf for the given frame indices
          # e.g. we use this to mask the frames around h_t when we have h_t in the readout in order to force
          # the attention to focus on the other frames
          if mask_att_opts is not None:
            enc_spatial_sizes = rf.copy_to_device(enc_spatial_dim.dyn_size_ext)
            mask_frame_idx = mask_att_opts["frame_idx"]  # [B]
            # just some heuristic
            # i.e. we mask:
            # 1 frame, if there are less than 6 frames
            # 3 frames, if there are less than 9 frames
            # 5 frames, otherwise
            mask_dimension = (enc_spatial_sizes // 3) * 2 - 1
            mask_dimension = rf.clip_by_value(mask_dimension, 1, 5)
            mask_dim = Dim(dimension=mask_dimension, name="att_mask")  # [5]
            # example mask: [-inf, -inf, -inf, 0, 0]
            mask_range = rf.range_over_dim(mask_dim)
            mask = rf.where(
              mask_range < mask_dimension,
              float("-inf"),
              0.0
            )
            # move mask to correct position along the time axis
            mask_indices = mask_range + mask_frame_idx - (mask_dimension // 2)  # [B, 5]
            mask_indices.sparse_dim = enc_spatial_dim
            mask_indices = rf.clip_by_value(
              mask_indices,
              0,
              enc_spatial_sizes - 1
            )
            mask = rf.scatter(
              mask,
              indices=mask_indices,
              indices_dim=mask_dim,
            )  # [B, T]
            energy = energy + mask

          att_weights = rf.softmax(energy, axis=enc_spatial_dim)
          att_weights = rf.dropout(att_weights, drop_prob=self.att_weight_dropout, axis=None)
        if hard_att_opts is not None and rf.get_run_ctx().epoch <= hard_att_opts["until_epoch"] + hard_att_opts["num_interpolation_epochs"]:
          if hard_att_opts["frame"] == "middle":
            frame_idx = rf.copy_to_device(enc_spatial_dim.dyn_size_ext) // 2
          else:
            frame_idx = rf.constant(hard_att_opts["frame"], dims=enc_spatial_dim.dyn_size_ext.dims)
          frame_idx.sparse_dim = enc_spatial_dim
          one_hot = rf.one_hot(frame_idx)
          one_hot = rf.expand_dim(one_hot, dim=self.att_num_heads)

          if rf.get_run_ctx().epoch <= hard_att_opts["until_epoch"] and hard_att_opts["frame"] == "middle":
            att_weights = one_hot
          else:
            interpolation_factor = (rf.get_run_ctx().epoch - hard_att_opts["until_epoch"]) / hard_att_opts["num_interpolation_epochs"]
            att_weights = (1 - interpolation_factor) * one_hot + interpolation_factor * att_weights

        if self.use_weight_feedback:
          state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5

        att = self.get_att(att_weights, enc, enc_spatial_dim)

    state_.att = att
    output_dict["att"] = att

    return output_dict, state_


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
          enc_ctx: Optional[rf.Tensor],  # not needed in case of trafo_att
          enc_spatial_dim: Dim,
          s: rf.Tensor,
          input_embed: Optional[rf.Tensor] = None,
          input_embed_spatial_dim: Optional[Dim] = None,
          use_mini_att: bool = False,
  ) -> rf.Tensor:
    if use_mini_att:
      assert input_embed is not None
      if "lstm" in self.decoder_state:
        assert input_embed_spatial_dim is not None
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
      if self.trafo_att:
        batch_dims = enc.remaining_dims(
          remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
        )
        att, _ = self.trafo_att(
          input_embed,
          spatial_dim=input_embed_spatial_dim,
          encoder=None if self.use_trafo_att_wo_cross_att else self.trafo_att.transform_encoder(enc, axis=enc_spatial_dim),
          state=self.trafo_att.default_initial_state(batch_dims=batch_dims)
        )
      else:
        s_transformed = self.s_transformed(s)

        weight_feedback = rf.zeros((self.enc_key_total_dim,))

        energy_in = enc_ctx + weight_feedback + s_transformed
        energy = self.energy(rf.tanh(energy_in))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        # we do not need use_mask because the softmax output is already padded with zeros
        att = self.get_att(att_weights, enc, enc_spatial_dim)

    return att
