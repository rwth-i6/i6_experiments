from typing import Optional, Dict, Any, Sequence, Tuple, List

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23.dependencies.returnn.network_builder_rf.base import BaseModel

from returnn.returnn.tensor import Tensor, Dim, single_step_dim
import returnn.returnn.frontend as rf
from returnn.returnn.frontend.encoder.conformer import ConformerEncoder, ConformerConvSubsample


class GlobalAttentionModel(BaseModel):
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

  def loop_step(
    self,
    *,
    enc: rf.Tensor,
    enc_ctx: rf.Tensor,
    inv_fertility: rf.Tensor,
    enc_spatial_dim: Dim,
    input_embed: rf.Tensor,
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

    weight_feedback = self.weight_feedback(state.accum_att_weights)
    s_transformed = self.s_transformed(s)
    energy_in = enc_ctx + weight_feedback + s_transformed
    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=enc_spatial_dim)
    state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5
    att0 = rf.dot(att_weights, enc, reduce=enc_spatial_dim, use_mask=False)
    att0.feature_dim = self.encoder.out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.encoder.out_dim))
    state_.att = att

    return {"s": s, "att": att}, state_
