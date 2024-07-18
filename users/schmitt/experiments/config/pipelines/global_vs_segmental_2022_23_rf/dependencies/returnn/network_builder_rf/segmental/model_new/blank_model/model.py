from typing import Optional, Dict, Any, Sequence, Tuple, List
from abc import ABC, abstractmethod
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf


class BlankDecoderBase(rf.Module, ABC):
  def __init__(self):
    super(BlankDecoderBase, self).__init__()
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)

  @property
  @abstractmethod
  def _s(self) -> rf.LSTM:
    pass

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s_blank=self._s.default_initial_state(batch_dims=batch_dims),
      i=rf.zeros(batch_dims, dtype="int32"),
    )
    return state

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s_blank": Tensor(
        "s_blank",
        dims=batch_dims + [self._s.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1
      ),
    }

  @staticmethod
  def _get_am(enc: rf.Tensor, enc_spatial_dim: Dim, state: rf.State, spatial_dim) -> rf.Tensor:
    if spatial_dim == single_step_dim:
      i = state.i
      clip_to_valid = True
    else:
      i = rf.range_over_dim(spatial_dim)
      # do clipping here, because rf.gather complains that i.dims_set is not a superset of enc_spatial_dim.dims_set
      seq_lens = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, i.device)
      i = rf.where(i < seq_lens, i, seq_lens - 1)
      clip_to_valid = False

    return rf.gather(enc, axis=enc_spatial_dim, indices=i, clip_to_valid=clip_to_valid)

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return None


class BlankDecoderV1(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          align_target_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV1, self).__init__()
    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim

    self.target_embed = rf.Embedding(align_target_dim, self.length_model_embed_dim)
    self.s = rf.LSTM(
      encoder_out_dim + self.length_model_embed_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  @property
  def _s(self) -> rf.LSTM:
    return self.s

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          state: rf.State,
          spatial_dim=single_step_dim
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    state_ = rf.State()

    am = self._get_am(enc, enc_spatial_dim, state, spatial_dim)
    s_blank, state_.s_blank = self.s(
      rf.concat_features(am, input_embed, allow_broadcast=True),
      state=state.s_blank,
      spatial_dim=spatial_dim
    )

    state_.i = state.i + 1

    return {"s_blank": s_blank}, state_

  def decode_logits(self, *, s_blank: Tensor) -> Tensor:
    """logits for the decoder"""
    logits = self.emit_prob(s_blank)
    return logits


class BlankDecoderV2(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          target_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV2, self).__init__()
    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim

    self.target_embed = rf.Embedding(target_dim, self.length_model_embed_dim)
    self.s = rf.LSTM(
      encoder_out_dim + length_model_embed_dim + label_state_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  @property
  def _s(self) -> rf.LSTM:
    return self.s

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          label_model_state: rf.Tensor,
          state: rf.State,
          spatial_dim=single_step_dim
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    state_ = rf.State()

    am = self._get_am(enc, enc_spatial_dim, state, spatial_dim)
    s_blank, state_.s_blank = self.s(
      rf.concat_features(am, input_embed, label_model_state),
      state=state.s_blank,
      spatial_dim=spatial_dim
    )

    state_.i = state.i + 1

    return {"s_blank": s_blank}, state_

  def decode_logits(self, *, s_blank: Tensor) -> Tensor:
    """logits for the decoder"""
    logits = self.emit_prob(s_blank)
    return logits

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV3(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV3, self).__init__()
    self.length_model_state_dim = length_model_state_dim

    self.s = rf.LSTM(
      encoder_out_dim + label_state_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  @property
  def _s(self) -> rf.LSTM:
    return self.s

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_spatial_dim: Dim,
          label_model_state: rf.Tensor,
          state: rf.State,
          spatial_dim=single_step_dim
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    state_ = rf.State()

    am = self._get_am(enc, enc_spatial_dim, state, spatial_dim)
    s_blank, state_.s_blank = self.s(
      rf.concat_features(am, label_model_state),
      state=state.s_blank,
      spatial_dim=spatial_dim
    )

    state_.i = state.i + 1

    return {"s_blank": s_blank}, state_

  def decode_logits(self, *, s_blank: Tensor) -> Tensor:
    """logits for the decoder"""
    logits = self.emit_prob(s_blank)
    return logits

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV4(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV4, self).__init__()
    self.length_model_state_dim = length_model_state_dim

    self.s = rf.Linear(
      encoder_out_dim + label_state_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim // 2, self.emit_prob_dim)

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State()
    return state

  @property
  def _s(self) -> rf.LSTM:
    raise NotImplementedError

  def decode_logits(self, *, enc: Tensor, label_model_states_unmasked: rf.Tensor, allow_broadcast: bool = False) -> Tensor:
    """logits for the decoder"""

    s_blank = self.s(rf.concat_features(enc, label_model_states_unmasked, allow_broadcast=allow_broadcast))
    s_blank = rf.reduce_out(s_blank, mode="max", num_pieces=2, out_dim=self.emit_prob.in_dim)
    s_blank = rf.dropout(s_blank, drop_prob=0.3, axis=rf.dropout_broadcast_default() and s_blank.feature_dim)
    logits = self.emit_prob(s_blank)
    return logits

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV5(BlankDecoderBase):
  @property
  def _s(self) -> rf.LSTM:
    raise NotImplementedError

  def __init__(
          self,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV5, self).__init__()

    self.emit_prob = rf.Linear(encoder_out_dim + label_state_dim, self.emit_prob_dim)

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State()
    return state

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV6(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV6, self).__init__()
    self.length_model_state_dim = length_model_state_dim

    self.s = rf.LSTM(
      encoder_out_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim + label_state_dim, self.emit_prob_dim)

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s_blank=self._s.default_initial_state(batch_dims=batch_dims),
    )
    return state

  @property
  def _s(self) -> rf.LSTM:
    return self.s

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV7(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
          distance_embed_dim: int,
          max_distance: int,
  ):
    super(BlankDecoderV7, self).__init__()
    self.length_model_state_dim = length_model_state_dim

    distance_embed_dim = Dim(name="distance_embed", dimension=distance_embed_dim)
    # + 1 for all distances > max_distance
    self.distance_dim = Dim(name="distance", dimension=max_distance + 1)

    self.distance_embed = rf.Embedding(self.distance_dim, distance_embed_dim)
    self.s = rf.Linear(
      encoder_out_dim + label_state_dim + distance_embed_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim // 2, self.emit_prob_dim)

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State()
    return state

  @property
  def _s(self) -> rf.LSTM:
    raise NotImplementedError

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]

  def decode_logits(
          self, *, enc: Tensor, label_model_states_unmasked: rf.Tensor, prev_emit_distances: rf.Tensor) -> Tensor:
    """logits for the decoder"""

    distance_embed = self.distance_embed(prev_emit_distances)

    s_blank = self.s(rf.concat_features(enc, label_model_states_unmasked, distance_embed))
    s_blank = rf.reduce_out(s_blank, mode="max", num_pieces=2, out_dim=self.emit_prob.in_dim)
    s_blank = rf.dropout(s_blank, drop_prob=0.3, axis=rf.dropout_broadcast_default() and s_blank.feature_dim)
    logits = self.emit_prob(s_blank)
    return logits
