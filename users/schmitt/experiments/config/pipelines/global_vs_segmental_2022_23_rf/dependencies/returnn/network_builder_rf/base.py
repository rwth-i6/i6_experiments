from typing import Optional, Dict, Any, Sequence, Tuple, List, Union, TYPE_CHECKING
import contextlib
import math
import functools

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf

from i6_experiments.users.schmitt.returnn_frontend.model_interfaces.supports_label_scorer_torch import RFModelWithMakeLabelScorer

_log_mel_feature_dim = 80
_batch_size_factor = 160


class LinearDecoder(rf.Module):
  def __init__(
          self,
          in_dim: Dim,
          out_dim: int,
  ):
    super(LinearDecoder, self).__init__()

    self.linear1 = rf.Linear(in_dim, Dim(name="ilm-linear1", dimension=out_dim))
    self.linear2 = rf.Linear(self.linear1.out_dim, Dim(name="ilm-linear2", dimension=out_dim))

    self.out_dim = self.linear2.out_dim

  def __call__(self, source: Tensor) -> Tensor:
    x = self.linear1(source)
    x = rf.dropout(x, drop_prob=0.1, axis=rf.dropout_broadcast_default() and x.feature_dim)
    x = rf.tanh(x)

    x = self.linear2(x)
    x = rf.dropout(x, drop_prob=0.1, axis=rf.dropout_broadcast_default() and x.feature_dim)
    x = rf.tanh(x)

    return x


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
          separate_blank_from_softmax: bool = False,
          reset_eos_params: bool = False,
          use_current_frame_in_readout: bool = False,
          target_embed_dim: Dim = Dim(name="target_embed", dimension=640),
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

    target_embed_opts = {"in_dim": target_dim, "out_dim": target_embed_dim}
    if reset_eos_params:
      self.target_embed_reset_eos = rf.Embedding(**target_embed_opts)
      self.target_embed = self.target_embed_reset_eos
    else:
      self.target_embed = rf.Embedding(**target_embed_opts)

    self.use_att_ctx_in_state = use_att_ctx_in_state
    self.use_weight_feedback = use_weight_feedback
    self.separate_blank_from_softmax = separate_blank_from_softmax

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
      assert decoder_state == "nb-2linear-ctx1"
      ilm_layer_class = LinearDecoder
      ilm_layer_opts = dict(
        out_dim=1024,
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
      if "lstm" in decoder_state:
        self.mini_att_lstm = rf.LSTM(self.target_embed.out_dim, Dim(name="mini-att-lstm", dimension=50))
        out_dim = self.mini_att_lstm.out_dim
      else:
        self.mini_att_linear = rf.Linear(self.target_embed.out_dim, Dim(name="mini-att-linear", dimension=50))
        out_dim = self.mini_att_linear.out_dim
      self.mini_att = rf.Linear(out_dim, self.att_num_heads * self.enc_out_dim)

    if use_weight_feedback:
      self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)

    self.s_transformed = rf.Linear(self.get_lstm().out_dim, enc_key_total_dim, with_bias=False)
    self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)

    readout_in_dim = self.get_lstm().out_dim + self.target_embed.out_dim + att_num_heads * enc_out_dim
    readout_out_dim = Dim(name="readout", dimension=1024)
    self.use_current_frame_in_readout = use_current_frame_in_readout
    if use_current_frame_in_readout:
      readout_in_dim += enc_out_dim
      self.readout_in_w_current_frame = rf.Linear(
        readout_in_dim,
        readout_out_dim,
      )
    else:
      self.readout_in = rf.Linear(
        readout_in_dim,
        readout_out_dim,
      )

    output_prob_opts = {"in_dim": readout_out_dim // 2, "out_dim": target_dim}
    if reset_eos_params:
      self.output_prob_reset_eos = rf.Linear(**output_prob_opts)
      self.output_prob = self.output_prob_reset_eos
    else:
      self.output_prob = rf.Linear(**output_prob_opts)

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
