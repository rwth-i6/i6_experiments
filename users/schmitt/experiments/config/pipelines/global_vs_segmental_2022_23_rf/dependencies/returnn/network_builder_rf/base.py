from typing import Optional, Dict, Any, Sequence, Tuple, List, Union, TYPE_CHECKING
import contextlib
import math
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.supports_label_scorer_torch import RFModelWithMakeLabelScorer

_log_mel_feature_dim = 80
_batch_size_factor = 160


class BaseLabelDecoder(rf.Module):
  def __init__(
          self,
          *,
          enc_out_dim: Dim,
          target_dim: Dim,
          blank_idx: int,
          enc_key_total_dim: Dim = Dim(name="enc_key_total_dim", dimension=1024),
          att_num_heads: Dim = Dim(name="att_num_heads", dimension=1),
          att_dropout: float = 0.1,
          l2: float = 0.0001,
          use_weight_feedback: bool = True,
          use_att_ctx_in_state: bool = True,
          decoder_state: str = "nb-lstm",
          use_mini_att: bool = False,
  ):
    super(BaseLabelDecoder, self).__init__()

    self.target_dim = target_dim
    self.blank_idx = blank_idx
    self.enc_out_dim = enc_out_dim

    self.enc_key_total_dim = enc_key_total_dim
    self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
    self.att_num_heads = att_num_heads
    self.att_dropout = att_dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()

    self.target_embed = rf.Embedding(target_dim, Dim(name="target_embed", dimension=640))

    self.use_att_ctx_in_state = use_att_ctx_in_state
    self.use_weight_feedback = use_weight_feedback

    self.decoder_state = decoder_state
    if "lstm" in decoder_state:
      ilm_layer_class = rf.ZoneoutLSTM
      ilm_layer_opts = dict(
        out_dim=Dim(name="lstm", dimension=1024),
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
          self.target_embed.out_dim + att_num_heads * enc_out_dim,
          **ilm_layer_opts,
        )
      else:
        self.s_wo_att = ilm_layer_class(
          self.target_embed.out_dim,
          **ilm_layer_opts,
        )
    else:
      ilm_layer_class = rf.Linear
      ilm_layer_opts = dict(
        out_dim=Dim(name="linear-ilm", dimension=1024),
      )
      if use_att_ctx_in_state:
        self.s_linear = ilm_layer_class(
          self.target_embed.out_dim + att_num_heads * enc_out_dim,
          **ilm_layer_opts,
        )
      else:
        self.s_wo_att_linear = ilm_layer_class(
          self.target_embed.out_dim,
          **ilm_layer_opts,
        )

    self.use_mini_att = use_mini_att
    if use_mini_att:
      self.mini_att_lstm = rf.LSTM(self.target_embed.out_dim, Dim(name="mini-att-lstm", dimension=50))
      self.mini_att = rf.Linear(self.mini_att_lstm.out_dim, self.att_num_heads * self.enc_out_dim)

    if use_weight_feedback:
      self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)

    self.s_transformed = rf.Linear(self.get_lstm().out_dim, enc_key_total_dim, with_bias=False)
    self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
    self.readout_in = rf.Linear(
      self.get_lstm().out_dim + self.target_embed.out_dim + att_num_heads * enc_out_dim,
      Dim(name="readout", dimension=1024),
    )
    self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

    for p in self.parameters():
      p.weight_decay = l2

    # Note: Even though we have this here, it is not used in loop_step or decode_logits.
    # Instead, it is intended to make a separate label scorer for it.
    self.language_model = None
    self.language_model_make_label_scorer = None

  def _update_state(
          self,
          input_embed: rf.Tensor,
          prev_att: rf.Tensor,
          prev_s_state: Optional[rf.LstmState],
  ) -> Tuple[rf.Tensor, Optional[rf.LstmState]]:
    if "lstm" in self.decoder_state:
      ilm_forward_opts = dict(
        state=prev_s_state,
        spatial_dim=single_step_dim,
      )
      if self.use_att_ctx_in_state:
        return self.s(rf.concat_features(input_embed, prev_att), **ilm_forward_opts)
      else:
        return self.s_wo_att(input_embed, **ilm_forward_opts)
    else:
      if self.use_att_ctx_in_state:
        return self.s_linear(rf.concat_features(input_embed, prev_att)), None
      else:
        return self.s_wo_att_linear(input_embed), None

  def get_lstm(self):
    if "lstm" in self.decoder_state:
      if self.use_att_ctx_in_state:
        return self.s
      else:
        return self.s_wo_att
    else:
      if self.use_att_ctx_in_state:
        return self.s_linear
      else:
        return self.s_wo_att_linear

  def get_att(
          self,
          att_weights: rf.Tensor,
          enc: rf.Tensor,
          reduce_dim: Dim
  ) -> rf.Tensor:
    att0 = rf.dot(att_weights, enc, reduce=reduce_dim, use_mask=False)
    att0.feature_dim = self.enc_out_dim
    att, _ = rf.merge_dims(att0, dims=(self.att_num_heads, self.enc_out_dim))
    return att
