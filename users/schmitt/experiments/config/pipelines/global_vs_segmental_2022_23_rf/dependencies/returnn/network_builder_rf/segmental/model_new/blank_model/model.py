from typing import Optional, Dict, Any, Sequence, Tuple, List
from abc import ABC, abstractmethod
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseLabelDecoder


class BlankDecoderBase(rf.Module, ABC):
  def __init__(self, length_model_state_dim: Dim):
    super(BlankDecoderBase, self).__init__()
    self.length_model_state_dim = length_model_state_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

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

  def decode_logits(self, *, s_blank: Tensor) -> Tensor:
    """logits for the decoder"""
    logits = self.emit_prob(s_blank)
    return logits

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
    super(BlankDecoderV1, self).__init__(length_model_state_dim=length_model_state_dim)
    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)

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
      rf.concat_features(am, input_embed),
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
    super(BlankDecoderV2, self).__init__(length_model_state_dim=length_model_state_dim)
    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)

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

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]


class BlankDecoderV3(BlankDecoderBase):
  def __init__(
          self,
          length_model_state_dim: Dim,
          label_state_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoderV3, self).__init__(length_model_state_dim=length_model_state_dim)
    self.length_model_state_dim = length_model_state_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)

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

  def get_label_decoder_deps(self) -> Optional[List[str]]:
    return ["s"]
