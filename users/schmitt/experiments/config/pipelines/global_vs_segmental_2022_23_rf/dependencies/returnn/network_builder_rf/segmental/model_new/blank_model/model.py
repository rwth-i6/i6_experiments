from typing import Optional, Dict, Any, Sequence, Tuple, List
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.model import ModelDef
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import _batch_size_factor, _log_mel_feature_dim
from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.base import BaseLabelDecoder


class BlankDecoder(rf.Module):
  def __init__(
          self,
          length_model_state_dim: Dim,
          length_model_embed_dim: Dim,
          align_target_dim: Dim,
          encoder_out_dim: Dim,
  ):
    super(BlankDecoder, self).__init__()
    self.length_model_state_dim = length_model_state_dim
    self.length_model_embed_dim = length_model_embed_dim
    self.emit_prob_dim = Dim(name="emit_prob", dimension=1)

    self.target_embed = rf.Embedding(align_target_dim, self.length_model_embed_dim)
    self.s = rf.LSTM(
      encoder_out_dim + self.length_model_embed_dim,
      self.length_model_state_dim,
    )
    self.emit_prob = rf.Linear(self.length_model_state_dim, self.emit_prob_dim)

  def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
    """Default initial state"""
    state = rf.State(
      s_blank=self.s.default_initial_state(batch_dims=batch_dims),
      i=rf.zeros(batch_dims, dtype="int32"),
    )
    return state

  def loop_step_output_templates(self, batch_dims: List[Dim]) -> Dict[str, Tensor]:
    """loop step out"""
    return {
      "s_blank": Tensor(
        "s_blank",
        dims=batch_dims + [self.s.out_dim],
        dtype=rf.get_default_float_dtype(),
        feature_dim_axis=-1
      ),
    }

  def loop_step(
          self,
          *,
          enc: rf.Tensor,
          enc_spatial_dim: Dim,
          input_embed: rf.Tensor,
          state: Optional[rf.State] = None,
          spatial_dim=single_step_dim
  ) -> Tuple[Dict[str, rf.Tensor], rf.State]:
    """step of the inner loop"""
    if state is None:
      batch_dims = enc.remaining_dims(
        remove=(enc.feature_dim, enc_spatial_dim) if enc_spatial_dim != single_step_dim else (enc.feature_dim,)
      )
      state = self.default_initial_state(batch_dims=batch_dims)
    state_ = rf.State()

    if spatial_dim == single_step_dim:
      i = state.i
      clip_to_valid = True
    else:
      i = rf.range_over_dim(spatial_dim)
      # do clipping here, because rf.gather complains that i.dims_set is not a superset of enc_spatial_dim.dims_set
      seq_lens = rf.copy_to_device(enc_spatial_dim.dyn_size_ext, i.device)
      i = rf.where(i < seq_lens, i, seq_lens - 1)
      clip_to_valid = False

    am = rf.gather(enc, axis=enc_spatial_dim, indices=i, clip_to_valid=clip_to_valid)
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
