from typing import Optional, Tuple, Dict

from returnn.tensor import Tensor, Dim, single_step_dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder, TransformerDecoderLayer

from i6_experiments.users.schmitt.experiments.config.pipelines.global_vs_segmental_2022_23_rf.dependencies.returnn.network_builder_rf.segmental import utils

_batch_size_factor = 160


class TrafoAttention(TransformerDecoder):
  def __init__(self, **kwargs):
    super(TrafoAttention, self).__init__(**kwargs)

    delattr(self, "input_embedding")
    delattr(self, "logits")

  def __call__(
          self,
          input_embeddings: Tensor,
          *,
          spatial_dim: Dim,
          state: rf.State,
          encoder: Optional[rf.State] = None,
          collected_outputs: Optional[Dict[str, Tensor]] = None,
  ) -> Tuple[Tensor, rf.State]:
    """
    forward, single step or whole sequence.

    :param source: labels
    :param spatial_dim: single_step_dim or spatial dim of source
    :param state: e.g. via :func:`default_initial_state`
    :param encoder: via :func:`transform_encoder`
    :param collected_outputs:
    :return: logits, new state
    """
    new_state = rf.State()

    decoded = input_embeddings * self.input_embedding_scale
    decoded = decoded + self.pos_enc(spatial_dim=spatial_dim, offset=state.pos)
    decoded = rf.dropout(decoded, self.input_dropout)
    if self.input_embedding_proj is not None:
      decoded = self.input_embedding_proj(decoded)

    new_state.pos = state.pos + (1 if spatial_dim == single_step_dim else spatial_dim.get_size_tensor())

    for layer_name, layer in self.layers.items():
      layer: TransformerDecoderLayer  # or similar
      decoded, new_state[layer_name] = layer(
        decoded,
        spatial_dim=spatial_dim,
        state=state[layer_name],
        encoder=encoder[layer_name] if encoder else None,
      )
      if collected_outputs is not None:
        collected_outputs[layer_name] = decoded

    decoded = self.final_layer_norm(decoded)

    return decoded, new_state


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
          use_current_frame_in_readout_w_gate: bool = False,
          use_current_frame_in_readout_random: bool = False,
          target_embed_dim: Dim = Dim(name="target_embed", dimension=640),
          target_embed_dropout: float = 0.0,
          att_weight_dropout: float = 0.0,
          use_hard_attention: bool = False,
          use_trafo_attention: bool = False,
  ):
    super(BaseLabelDecoder, self).__init__()

    assert not (use_current_frame_in_readout and use_current_frame_in_readout_w_gate), "only one of them allowed"

    self.target_dim = target_dim
    self.blank_idx = blank_idx
    self.enc_out_dim = enc_out_dim

    self.enc_key_total_dim = enc_key_total_dim
    self.enc_key_per_head_dim = enc_key_total_dim.div_left(att_num_heads)
    self.att_num_heads = att_num_heads
    self.att_dropout = att_dropout
    self.dropout_broadcast = rf.dropout_broadcast_default()
    self.target_embed_dropout = target_embed_dropout
    self.att_weight_dropout = att_weight_dropout

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

    if not use_hard_attention and not use_trafo_attention:
      self.s_transformed = rf.Linear(self.get_lstm().out_dim, enc_key_total_dim, with_bias=False)
      self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
      att_dim = att_num_heads * enc_out_dim
      self.trafo_att = None
    else:
      if use_trafo_attention:
        model_dim = target_embed_dim
        self.trafo_att = TrafoAttention(
          num_layers=6,
          encoder_dim=enc_out_dim,
          vocab_dim=target_dim,  # not used (embeddings are calculated before)
          model_dim=model_dim,
          sequential=rf.Sequential,
          share_embedding=True,  # not used
          input_embedding_scale=model_dim.dimension ** 0.5  # not used
        )
        att_dim = model_dim
      else:
        self.trafo_att = None
        att_dim = att_num_heads * enc_out_dim

    readout_in_dim = self.get_lstm().out_dim + self.target_embed.out_dim + att_dim
    readout_out_dim = Dim(name="readout", dimension=1024)
    self.use_current_frame_in_readout = use_current_frame_in_readout
    if use_current_frame_in_readout:
      readout_in_dim += enc_out_dim
      self.readout_in_w_current_frame = rf.Linear(
        readout_in_dim,
        readout_out_dim,
      )
      self.readout_in = self.readout_in_w_current_frame
    else:
      self.readout_in = rf.Linear(
        readout_in_dim,
        readout_out_dim,
      )

    self.use_current_frame_in_readout_w_gate = use_current_frame_in_readout_w_gate
    if use_current_frame_in_readout_w_gate:
      self.attention_gate = rf.Linear(
        att_num_heads * enc_out_dim,
        Dim(name="attention_gate", dimension=1),
      )

    self.use_current_frame_in_readout_random = use_current_frame_in_readout_random

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

  def decode_logits(self, *, s: Tensor, input_embed: Tensor, att: Tensor, h_t: Optional[Tensor] = None) -> Tensor:
    input_embed = rf.dropout(input_embed, drop_prob=self.target_embed_dropout, axis=None)

    if self.use_current_frame_in_readout:
      assert h_t is not None, "Need h_t for readout!"
      readout_input = rf.concat_features(s, input_embed, att, h_t, allow_broadcast=True)
    elif self.use_current_frame_in_readout_w_gate or self.use_current_frame_in_readout_random:
      if not (not rf.get_run_ctx().train_flag and self.use_current_frame_in_readout_random):
        assert h_t is not None, "Need h_t for readout!"
      if self.use_current_frame_in_readout_w_gate:
        alpha = rf.sigmoid(self.attention_gate(att))
      else:
        alpha = rf.random_uniform(
          dims=att.remaining_dims(att.feature_dim),
          dtype="int32",
          minval=0,
          maxval=2,
        )
        alpha = rf.cast(alpha, "float32")
      # need to have same feature dim for interpolation
      if h_t is not None:
        h_t = utils.copy_tensor_replace_dim_tag(h_t, self.enc_out_dim, att.feature_dim)

      if self.use_current_frame_in_readout_random and not rf.get_run_ctx().train_flag:
        if h_t is None:
          att_h_t = att
        else:
          att_h_t = h_t
      else:
        att_h_t = alpha * att + (1 - alpha) * h_t
        if self.use_current_frame_in_readout_w_gate:
          att_h_t = rf.squeeze(att_h_t, axis=self.attention_gate.out_dim)
        att_h_t.feature_dim = att.feature_dim

      readout_input = rf.concat_features(s, input_embed, att_h_t, allow_broadcast=True)
    else:
      readout_input = rf.concat_features(s, input_embed, att, allow_broadcast=True)

    readout_in = self.readout_in(readout_input)
    readout = rf.reduce_out(readout_in, mode="max", num_pieces=2, out_dim=self.output_prob.in_dim)
    readout = rf.dropout(readout, drop_prob=0.3, axis=self.dropout_broadcast and readout.feature_dim)
    logits = self.output_prob(readout)
    return logits


def get_common_config_params():
  from returnn.config import get_global_config

  config = get_global_config()  # noqa

  enc_aux_logits = config.typed_value("aux_loss_layers")
  pos_emb_dropout = config.float("pos_emb_dropout", 0.0)
  log_mel_feature_dim = config.int("log_mel_feature_dim", 80)
  # real input is raw audio, internally it does logmel
  in_dim = Dim(name="logmel", dimension=log_mel_feature_dim, kind=Dim.Types.Feature)
  language_model = config.typed_value("external_lm")
  use_weight_feedback = config.bool("use_weight_feedback", True)
  use_att_ctx_in_state = config.bool("use_att_ctx_in_state", True)
  label_decoder_state = config.typed_value("label_decoder_state", "nb-lstm")
  target_embed_dim = config.int("target_embed_dim", 640)
  feature_extraction_opts = config.typed_value("feature_extraction_opts", None)
  use_mini_att = config.bool("use_mini_att", False)
  att_dropout = config.float("att_dropout", 0.1)
  target_embed_dropout = config.float("target_embed_dropout", 0.0)
  att_weight_dropout = config.float("att_weight_dropout", 0.0)

  return dict(
    in_dim=in_dim,
    enc_aux_logits=enc_aux_logits or (),
    pos_emb_dropout=pos_emb_dropout,
    language_model=language_model,
    use_weight_feedback=use_weight_feedback,
    use_att_ctx_in_state=use_att_ctx_in_state,
    label_decoder_state=label_decoder_state,
    target_embed_dim=target_embed_dim,
    feature_extraction_opts=feature_extraction_opts,
    use_mini_att=use_mini_att,
    att_dropout=att_dropout,
    target_embed_dropout=target_embed_dropout,
    att_weight_dropout=att_weight_dropout,
  )


def apply_weight_dropout(model: rf.Module):
  from returnn.config import get_global_config
  config = get_global_config()  # noqa

  weight_dropout = config.float("weight_dropout", None)
  if weight_dropout:
    # Use some blacklist. I think the same blacklist as for weight decay is reasonable.
    # Usually sth like: ["rf.Embedding", "rf.LearnedRelativePositionalEncoding"]
    blacklist = config.typed_value("optimizer")["weight_decay_modules_blacklist"]
    blacklist = tuple(eval(name, {"rf": rf}) for name in blacklist)
    for mod in model.modules():
      if isinstance(mod, blacklist):
        continue
      for param_name, param in mod.named_parameters(recurse=False):
        if param_name.endswith("bias"):  # no bias
          continue
        if param.auxiliary:
          continue
        rf.weight_dropout(mod, param_name, drop_prob=weight_dropout)
