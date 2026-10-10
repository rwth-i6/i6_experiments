from typing import Optional

import returnn.frontend as rf
from returnn.tensor import Dim as ReturnnDim
from returnn.tensor import Tensor as ReturnnTensor

_group_cross_attention_class = None


def group_cross_attention_class():
    """
    Builds the upstream chunk-masked cross-attention with its attention stated to the frontend, once.

    The upstream module computes the energies of every frame and masks the ones outside the query's
    chunks. This one hands the chunk of every query and of every frame to :func:`rf.dot_attention`, which
    then reads only the frames a query attends. Same parameters, same numbers.

    :return: the attention class, a subclass of ChunkMaskedCrossAttention
    """
    global _group_cross_attention_class
    if _group_cross_attention_class is not None:
        return _group_cross_attention_class
    from i6_experiments.users.zeyer.nn_rf.decoder.streaming.cross_attn import ChunkMaskedCrossAttention

    class ChunkGroupCrossAttention(ChunkMaskedCrossAttention):
        """Attends through the frontend with the chunks as groups, the decoder sets the mode on every instance."""

        group_mode = "less_equal"
        max_group_size: Optional[int] = None

        def __call__(
            self,
            query: ReturnnTensor,
            *,
            keys: ReturnnTensor,
            values: ReturnnTensor,
            enc_spatial_dim: ReturnnDim,
            query_chunk_idx: ReturnnTensor,
            key_chunk_idx: ReturnnTensor,
        ) -> ReturnnTensor:
            q = self.q_proj(query)
            q = rf.split_dims(q, axis=self.key_dim_total, dims=[self.num_heads, self.key_dim_per_head])
            att = rf.dot_attention(
                q,
                keys,
                values,
                key_dim=self.key_dim_per_head,
                axis=enc_spatial_dim,
                att_dropout=self.att_dropout,
                att_dropout_broadcast=False,
                query_group=query_chunk_idx,
                key_group=key_chunk_idx,
                group_mode=self.group_mode,
                max_group_size=self.max_group_size,
            )
            att, _ = rf.merge_dims(att, dims=[self.num_heads, self.value_dim_per_head], out_dim=self.value_dim_total)
            return self.proj(att)

    _group_cross_attention_class = ChunkGroupCrossAttention
    return _group_cross_attention_class
