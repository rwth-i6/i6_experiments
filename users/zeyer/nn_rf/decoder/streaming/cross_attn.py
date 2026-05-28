"""
Chunk- (and frame-) masked cross-attention for streaming decoders.

The decoder runs over an augmented label sequence; each decoder position belongs
to a chunk ``q_chunk``. Cross-attention to the encoder is restricted to encoder
frames whose chunk index is ``<= q_chunk`` (all frames causally available by the
end of that chunk). The mask is additive (-inf on disallowed) and applied to the
attention energy before the softmax over the encoder time axis.

This is the shared primitive for chunkwise.py (and reused by the others). For the
frame-synchronous variant, pass per-frame indices instead of chunk indices -- the
``<=`` semantics are identical.
"""

from __future__ import annotations

from typing import Optional, Tuple
import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


class ChunkMaskedCrossAttention(rf.Module):
    """Multi-head cross-attention with a ``key_chunk <= query_chunk`` causal-by-chunk mask."""

    def __init__(
        self,
        encoder_dim: Dim,
        query_in_dim: Dim,
        *,
        key_dim_total: Dim,
        value_dim_total: Dim,
        num_heads: int,
        with_bias: bool = False,
        att_dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.query_in_dim = query_in_dim
        if isinstance(num_heads, int):
            num_heads = Dim(num_heads, name="att_heads")
        self.num_heads = num_heads
        self.key_dim_total = key_dim_total
        self.key_dim_per_head = key_dim_total.div_left(num_heads)
        self.value_dim_total = value_dim_total
        self.value_dim_per_head = value_dim_total.div_left(num_heads)
        self.att_dropout = att_dropout

        self.q_proj = rf.Linear(query_in_dim, key_dim_total, with_bias=with_bias)
        self.kv_proj = rf.Linear(encoder_dim, key_dim_total + value_dim_total, with_bias=with_bias)
        self.proj = rf.Linear(value_dim_total, query_in_dim, with_bias=with_bias)

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> Tuple[Tensor, Tensor]:
        """Precompute keys and values from the encoder output (done once per seq)."""
        kv = self.kv_proj(encoder)
        k, v = rf.split(kv, axis=kv.feature_dim, out_dims=[self.key_dim_total, self.value_dim_total])
        k = rf.split_dims(k, axis=self.key_dim_total, dims=[self.num_heads, self.key_dim_per_head])
        v = rf.split_dims(v, axis=self.value_dim_total, dims=[self.num_heads, self.value_dim_per_head])
        return k, v

    def __call__(
        self,
        query: Tensor,
        *,
        keys: Tensor,
        values: Tensor,
        enc_spatial_dim: Dim,
        query_chunk_idx: Tensor,
        key_chunk_idx: Tensor,
    ) -> Tensor:
        """
        :param query: {..., query_in_dim}, with whatever decoder spatial/step axis.
        :param keys: {..., enc_spatial_dim, num_heads, key_dim_per_head} (from transform_encoder)
        :param values: {..., enc_spatial_dim, num_heads, value_dim_per_head}
        :param enc_spatial_dim: encoder time axis to attend over.
        :param query_chunk_idx: int {..., query axis} -- chunk index of each query position.
        :param key_chunk_idx: int {enc_spatial_dim} -- chunk index of each encoder frame (= frame // chunk_size).
        :return: {..., query_in_dim}
        """
        q = self.q_proj(query)
        q = rf.split_dims(q, axis=self.key_dim_total, dims=[self.num_heads, self.key_dim_per_head])
        q *= self.key_dim_per_head.dimension**-0.5
        energy = rf.matmul(q, keys, reduce=self.key_dim_per_head)  # {..., q-axis, num_heads, enc_spatial}
        # chunk mask: allow key frame iff key_chunk_idx <= query_chunk_idx
        allowed = key_chunk_idx <= query_chunk_idx  # broadcasts over {q-axis, enc_spatial}
        energy = rf.where(allowed, energy, float("-inf"))
        att_weights = rf.softmax(energy, axis=enc_spatial_dim)
        att_weights = rf.dropout(att_weights, self.att_dropout, axis=False)
        att = rf.matmul(att_weights, values, reduce=enc_spatial_dim, use_mask=False)
        att.feature_dim = self.value_dim_per_head
        att, _ = rf.merge_dims(att, dims=[self.num_heads, self.value_dim_per_head], out_dim=self.value_dim_total)
        return self.proj(att)
