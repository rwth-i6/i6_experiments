from typing import Callable, Optional, Tuple, cast

import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoderLayer, make_norm
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


class ChunkCrossAttentionDecoderLayer(TransformerDecoderLayer):
    """Decoder layer whose cross-attention reads the encoder frames of the chunks up to the chunk of every position."""

    def __init__(
        self,
        encoder_dim: ReturnnDim,
        out_dim: ReturnnDim,
        *,
        num_heads: int,
        att_dropout: float,
        norm: Callable,
        **kwargs,
    ):
        """
        :param encoder_dim: the width of the encoder frames
        :param out_dim: the model dim
        :param num_heads: attention heads of the self-attention and the cross-attention
        :param att_dropout: dropout on the attention weights of both
        :param norm: the norm in front of every block
        :param kwargs: the other options of :class:`TransformerDecoderLayer`
        """
        super().__init__(None, out_dim, num_heads=num_heads, att_dropout=att_dropout, norm=norm, **kwargs)

        self.encoder_dim = encoder_dim
        self.cross_att = group_cross_attention_class()(
            encoder_dim,
            out_dim,
            key_dim_total=out_dim,
            value_dim_total=out_dim,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )
        self.cross_att_layer_norm = make_norm(norm, out_dim)

    def __call__(
        self, x: ReturnnTensor, *, spatial_dim: ReturnnDim, state: rf.State, encoder: Optional[rf.State] = None
    ) -> Tuple[ReturnnTensor, rf.State]:
        """
        Runs the plain self-attention, the cross-attention and the feed-forward.

        :param x: the layer input
        :param spatial_dim: the spatial dim, ``single_step_dim`` while decoding
        :param state: the self-attention state
        :param encoder: the input of the layer besides x, ``cross_att`` the keys and the values of the frames,
            ``enc_spatial_dim`` their dim, ``query_chunk_idx`` the chunk of every position, ``key_chunk_idx`` of every
            frame
        :return: (the layer output, the new state)
        """
        assert encoder is not None, "the layer takes the frames as its encoder input"
        new_state = rf.State()

        # the modules of the plain layer with their call types
        self_att = cast(rf.CausalSelfAttention, self.self_att)
        self_att_norm = cast(Callable[[ReturnnTensor], ReturnnTensor], self.self_att_layer_norm)
        cross_att_norm = cast(Callable[[ReturnnTensor], ReturnnTensor], self.cross_att_layer_norm)
        ff_norm = cast(Callable[[ReturnnTensor], ReturnnTensor], self.ff_layer_norm)
        ff = cast(Callable[[ReturnnTensor], ReturnnTensor], self.ff)

        # self-attention
        x_sa, new_state.self_att = self_att(self_att_norm(x), axis=spatial_dim, state=state.self_att)
        x_sa = rf.dropout(x_sa, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_sa + x

        # cross-attention
        keys, values = encoder.cross_att
        x_ca = cast(Callable[..., ReturnnTensor], self.cross_att)(
            cross_att_norm(x),
            keys=keys,
            values=values,
            enc_spatial_dim=encoder.enc_spatial_dim,
            query_chunk_idx=encoder.query_chunk_idx,
            key_chunk_idx=encoder.key_chunk_idx,
        )
        x_ca = rf.dropout(x_ca, self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_ca + x

        # feed-forward
        x_ff = rf.dropout(ff(ff_norm(x)), self.dropout, axis=self.dropout_broadcast and self.out_dim)
        x = x_ff + x

        return x, new_state
