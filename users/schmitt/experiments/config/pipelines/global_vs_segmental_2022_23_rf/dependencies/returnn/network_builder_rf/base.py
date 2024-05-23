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

    self.s = rf.ZoneoutLSTM(
      self.target_embed.out_dim + att_num_heads * enc_out_dim,
      Dim(name="lstm", dimension=1024),
      zoneout_factor_cell=0.15,
      zoneout_factor_output=0.05,
      use_zoneout_output=False,  # like RETURNN/TF ZoneoutLSTM old default
      # parts_order="icfo",  # like RETURNN/TF ZoneoutLSTM
      # parts_order="ifco",
      parts_order="jifo",  # NativeLSTM (the code above converts it...)
      forget_bias=0.0,  # the code above already adds it during conversion
    )

    self.weight_feedback = rf.Linear(att_num_heads, enc_key_total_dim, with_bias=False)
    self.s_transformed = rf.Linear(self.s.out_dim, enc_key_total_dim, with_bias=False)
    self.energy = rf.Linear(enc_key_total_dim, att_num_heads, with_bias=False)
    self.readout_in = rf.Linear(
      self.s.out_dim + self.target_embed.out_dim + att_num_heads * enc_out_dim,
      Dim(name="readout", dimension=1024),
    )
    self.output_prob = rf.Linear(self.readout_in.out_dim // 2, target_dim)

    for p in self.parameters():
      p.weight_decay = l2

    # Note: Even though we have this here, it is not used in loop_step or decode_logits.
    # Instead, it is intended to make a separate label scorer for it.
    self.language_model = None
    self.language_model_make_label_scorer = None
