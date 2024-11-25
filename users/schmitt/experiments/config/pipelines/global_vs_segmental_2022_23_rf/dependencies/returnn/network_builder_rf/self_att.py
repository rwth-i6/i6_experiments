import returnn.frontend as rf
from returnn.frontend import Tensor, Dim
from returnn.frontend.attention import RelPosSelfAttention, relative_positional_encoding, _rel_pos_enc_shift


class EpochConditionedRelPosSelfAttention(RelPosSelfAttention):
  def __init__(self, enable_from_epoch: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.enable_from_epoch = enable_from_epoch

  def __call__(self, source: Tensor, *, axis: Dim, **_kwargs) -> Tensor:
    """forward"""
    if self.learned_pos_emb is not None:
      pos_emb, pos_emb_spatial_dim = self.learned_pos_emb(query_spatial_dim=axis, key_value_spatial_dim=axis)
    else:
      pos_emb, pos_emb_spatial_dim = relative_positional_encoding(
        query_spatial_dim=axis, key_value_spatial_dim=axis, feat_dim=self.pos_emb_feat_dim
      )
    if self.pos_emb_dropout:
      pos_emb = rf.dropout(pos_emb, self.pos_emb_dropout)
    if self.linear_pos is not None:
      pos_emb = self.linear_pos(pos_emb)
    if self.separate_pos_emb_per_head:
      pos_emb = rf.split_dims(pos_emb, axis=self.key_dim_total, dims=(self.num_heads, self.key_dim_per_head))
    # pos_emb: (head, 2*time1-1, d_k)

    q, k, v = self.forward_qkv(source)

    # until enable_from_epoch is reached, just return the values instead of computing the attention
    if rf.get_run_ctx().epoch < self.enable_from_epoch:
      output, _ = rf.merge_dims(v, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
      return output

    hist_dim = Dim(None, name=f"{axis.description}:kv")
    k, _ = rf.replace_dim(k, in_dim=axis, out_dim=hist_dim)
    v, _ = rf.replace_dim(v, in_dim=axis, out_dim=hist_dim)
    q_with_bias_u = (q + self.pos_bias_u) if self.pos_bias_u is not None else q  # (batch, head, time1, d_k)
    q_with_bias_v = (q + self.pos_bias_v) if self.pos_bias_v is not None else q  # (batch, head, time1, d_k)

    # compute attention score
    # first compute matrix a and matrix c
    # as described in https://arxiv.org/abs/1901.02860 Section 3.3
    # (batch, head, time1, time2)
    matrix_ac = rf.matmul(q_with_bias_u, k, reduce=self.key_dim_per_head)

    # compute matrix b and matrix d
    # (batch, head, time1, 2*time1-1)
    matrix_bd = rf.matmul(q_with_bias_v, pos_emb, reduce=self.key_dim_per_head)
    matrix_bd = _rel_pos_enc_shift(matrix_bd, axis, pos_emb_spatial_dim, hist_dim)

    scores = matrix_ac + matrix_bd  # (batch, head, time1, time2)
    scores *= self.key_dim_per_head.dimension ** -0.5
    att_weights = rf.softmax(scores, axis=hist_dim)
    att_weights = rf.dropout(att_weights, self.att_dropout, axis=self.att_dropout_broadcast and hist_dim)
    # Masking not needed because softmax should already have masked,
    # so we have 0.0 att weights for padded frames.
    att = rf.matmul(att_weights, v, reduce=hist_dim, use_mask=False)
    output, _ = rf.merge_dims(att, dims=(self.num_heads, self.value_dim_per_head), out_dim=self.value_dim_total)
    if self.proj:
      output = self.proj(output)
    return output
