from typing import Sequence, List, Optional, Tuple, Dict

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils

_batch_size_factor = 160


class GlobalAttDecoder(rf.Module):
  def __init__(
          self,
          *,
          enc_out_dim: Dim,
          target_dim: Dim,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          att_dropout: float = 0.1,
          use_weight_feedback: bool = True,
          use_att_ctx_in_state: bool = True,
          decoder_state: str = "nb-lstm",
          target_embed_dim: Dim = Dim(name="target_embed", dimension=640),
          ilm_dimension: int = 1024,
          readout_dimension: int = 1024,
          target_embed_dropout: float = 0.0,
          att_weight_dropout: float = 0.0,
          use_mini_att: bool = False,
  ):
    super(GlobalAttDecoder, self).__init__()

    from returnn.config import get_global_config
    config = get_global_config()  # noqa

    self.target_dim = target_dim
    self.enc_out_dim = enc_out_dim

    self.enc_key_total_dim = enc_key_total_dim
    self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
    self.att_num_heads = att_num_heads
    self.att_dropout = att_dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()
    self.target_embed_dropout = target_embed_dropout
    self.att_weight_dropout = att_weight_dropout

    target_embed_opts = {"in_dim": target_dim, "out_dim": target_embed_dim}
    self.target_embed = rf.Embedding(**target_embed_opts)

    self.use_att_ctx_in_state = use_att_ctx_in_state
    self.use_weight_feedback = use_weight_feedback

    att_dim = att_num_heads * enc_out_dim
    self.att_dim = att_dim

    # predictor
    self.decoder_state = decoder_state
    ilm_layer_class = rf.ZoneoutLSTM
    ilm_layer_opts = dict(
      out_dim=Dim(name="lstm", dimension=ilm_dimension),
      zoneout_factor_cell=0.15,
      zoneout_factor_output=0.05,
      use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
      # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
      # parts_order="ifco",
      parts_order="jifo",  # NativeLSTM (the code above converts it...)
      forget_bias=0.0,  # the code above already adds it during conversion
    )
    if use_att_ctx_in_state:
      self.s = ilm_layer_class(
        self.target_embed.out_dim + att_dim,
        **ilm_layer_opts,
      )
    else:
      self.s_wo_att = ilm_layer_class(
        self.target_embed.out_dim,
        **ilm_layer_opts,
      )

    # attention
    self.enc_ctx = rf.Linear(enc_out_dim, enc_key_total_dim)

    if use_weight_feedback:
      self.inv_fertility = rf.Linear(enc_out_dim, att_num_heads, with_bias=False)
      self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)

    self.s_transformed = rf.Linear(self.get_predictor().out_dim, enc_key_total_dim, with_bias=False)
    self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)

    self.use_mini_att = use_mini_att
    if use_mini_att:
      mini_lstm_out_dimension = config.int("mini_lstm_out_dimension", 50)
      if "lstm" in decoder_state:
        self.mini_att_lstm = rf.LSTM(
          self.target_embed.out_dim, Dim(name="mini-att-lstm", dimension=mini_lstm_out_dimension))
        out_dim = self.mini_att_lstm.out_dim
      else:
        self.mini_att_linear = rf.Linear(
          self.target_embed.out_dim, Dim(name="mini-att-linear", dimension=mini_lstm_out_dimension))
        out_dim = self.mini_att_linear.out_dim
      self.mini_att = rf.Linear(out_dim, att_dim)

    # joiner
    readout_in_dim = self.get_predictor().out_dim
    readout_in_dim += self.target_embed.out_dim
    readout_in_dim += att_dim

    readout_out_dim = Dim(name="readout", dimension=readout_dimension)
    self.readout_in = rf.Linear(
      readout_in_dim,
      readout_out_dim,
    )

    # output prob
    output_prob_opts = {"in_dim": readout_out_dim // 2, "out_dim": target_dim}
    self.output_prob = rf.Linear(**output_prob_opts)

  def _update_state(
          self,
          input_embed: rf.Tensor,
          prev_att: rf.Tensor,
          prev_s_state: Optional[rf.LstmState],
  ) -> Tuple[rf.Tensor, Optional[rf.LstmState]]:
    ilm_forward_opts = dict(
      state=prev_s_state,
      spatial_dim=single_step_dim,
    )
    if self.use_att_ctx_in_state:
      return self.s(rf.concat_features(input_embed, prev_att), **ilm_forward_opts)
    else:
      return self.s_wo_att(input_embed, **ilm_forward_opts)

  def get_predictor(self):
    if self.use_att_ctx_in_state:
      return self.s
    else:
      return self.s_wo_att

  def get_att(
          self,
          att_weights: rf.Tensor,
          enc: rf.Tensor,
          reduce_dim: Dim
  ) -> rf.Tensor:
    att0 = rf.dot(att_weights, enc, reduce=reduce_dim, use_mask=False)
    att0.feature_dim = self.enc_out_dim
    if self.att_num_heads in att0.dims:
      att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.enc_out_dim))
    else:
      att = att0
      # change feature dim to num_heads*enc_out_dim (this is what we expect)
      att = utils.copy_tensor_replace_dim_tag(att, self.enc_out_dim, self.att_num_heads * self.enc_out_dim)
    return att

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

    state.att = rf.zeros(list(batch_dims) + [self.att_dim])
    state.att.feature_dim_axis = len(state.att.dims) - 1

    state.s = self.get_predictor().default_initial_state(batch_dims=batch_dims)
    if use_mini_att:
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
        "s", dims=batch_dims + [self.get_predictor().out_dim], dtype=rf.get_default_float_dtype(), feature_dim_axis=-1
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
    prev_s_state = state.s if "lstm" in self.decoder_state else None

    input_embed = rf.dropout(input_embed, drop_prob=self.target_embed_dropout, axis=None)
    s, s_state = self._update_state(input_embed, prev_att, prev_s_state)
    output_dict["s"] = s
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
      if self.use_weight_feedback:
        weight_feedback = self.weight_feedback(state.accum_att_weights)
      else:
        weight_feedback = rf.zeros((self.enc_key_total_dim,))

      s_transformed = self.s_transformed(s)
      energy_in = enc_ctx + weight_feedback + s_transformed

      energy = self.energy(rf.tanh(energy_in))

      att_weights = rf.softmax(energy, axis=enc_spatial_dim)
      att_weights = rf.dropout(att_weights, drop_prob=self.att_weight_dropout, axis=None)

      if self.use_weight_feedback:
        state_.accum_att_weights = state.accum_att_weights + att_weights * inv_fertility * 0.5

      att = self.get_att(att_weights, enc, enc_spatial_dim)

    state_.att = att
    output_dict["att"] = att

    return output_dict, state_

  def decode_logits(
          self,
          *,
          s: Tensor,
          input_embed: Tensor,
          att: Tensor,
  ) -> Tensor:
    readout_input = rf.concat_features(s, input_embed, att, allow_broadcast=True)
    readout_in = self.readout_in(readout_input)
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)

    return logits


class GlobalAttEfficientDecoder(GlobalAttDecoder):
  def __init__(self, **kwargs):
    super(GlobalAttEfficientDecoder, self).__init__(**kwargs)

    assert not self.use_att_ctx_in_state and not self.use_weight_feedback, (
      "Cannot have prev att dependency for efficient implementation!"
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
  ) -> rf.Tensor:
    s_transformed = self.s_transformed(s)

    weight_feedback = rf.zeros((self.enc_key_total_dim,))

    energy_in = enc_ctx + weight_feedback + s_transformed
    energy = self.energy(rf.tanh(energy_in))
    att_weights = rf.softmax(energy, axis=enc_spatial_dim)
    att = self.get_att(att_weights, enc, enc_spatial_dim)

    return att

