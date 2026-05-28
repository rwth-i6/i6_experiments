"""
Chunk-synchronous streaming decoder (slow-fast-rna project).

The decoder runs over the EOC-augmented label sequence produced by
:func:`...streaming.segmentation.chunk_augmented_targets`:
for each encoder chunk it emits that chunk's labels followed by an end-of-chunk
(EOC) marker. Self-attention is causal over the label sequence; cross-attention
is restricted (via :class:`...streaming.cross_attn.ChunkMaskedCrossAttention`) to
encoder frames in chunks ``<=`` the query position's chunk -- i.e. only audio that
has streamed in by the end of that chunk.

Training is teacher-forced over the whole augmented sequence in one pass.
Recog decodes chunk by chunk: within chunk k, emit labels autoregressively until
EOC, then advance to chunk k+1, stopping when the encoder chunks are exhausted.
"""

from __future__ import annotations

from typing import Optional, Any, Dict, Sequence, Tuple, List
import functools

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim

from .cross_attn import ChunkMaskedCrossAttention


class ChunkwiseDecoderLayer(rf.Module):
    def __init__(
        self,
        model_dim: Dim,
        encoder_dim: Dim,
        ff_dim: Dim,
        *,
        num_heads: int,
        dropout: float,
        att_dropout: float,
    ):
        super().__init__()
        self.self_att_ln = rf.LayerNorm(model_dim)
        self.self_att = rf.CausalSelfAttention(
            model_dim,
            proj_dim=model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )
        self.cross_att_ln = rf.LayerNorm(model_dim)
        self.cross_att = ChunkMaskedCrossAttention(
            encoder_dim,
            model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )
        self.ff_ln = rf.LayerNorm(model_dim)
        self.ff = _FeedForward(model_dim, ff_dim, dropout=dropout)
        self.dropout = dropout

    def __call__(
        self,
        x: Tensor,
        *,
        spatial_dim: Dim,
        self_att_state: rf.State,
        keys: Tensor,
        values: Tensor,
        enc_spatial_dim: Dim,
        query_chunk_idx: Tensor,
        key_chunk_idx: Tensor,
    ) -> Tuple[Tensor, rf.State]:
        h, new_state = self.self_att(self.self_att_ln(x), spatial_dim, state=self_att_state)
        x = x + rf.dropout(h, self.dropout, axis=h.feature_dim)
        h = self.cross_att(
            self.cross_att_ln(x),
            keys=keys,
            values=values,
            enc_spatial_dim=enc_spatial_dim,
            query_chunk_idx=query_chunk_idx,
            key_chunk_idx=key_chunk_idx,
        )
        x = x + rf.dropout(h, self.dropout, axis=h.feature_dim)
        x = x + self.ff(self.ff_ln(x))
        return x, new_state


class ChunkwiseDecoder(rf.Module):
    """Chunk-synchronous Transformer decoder with chunk-masked cross-attention."""

    def __init__(
        self,
        *,
        encoder_dim: Dim,
        vocab_dim: Dim,
        chunk_size: int,
        eoc_idx: int,
        model_dim: int = 512,
        ff_dim: int = 2048,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
    ):
        super().__init__()
        if isinstance(model_dim, int):
            model_dim = Dim(model_dim, name="dec_model")
        if isinstance(ff_dim, int):
            ff_dim = Dim(ff_dim, name="dec_ff")
        self.model_dim = model_dim
        self.vocab_dim = vocab_dim
        self.encoder_dim = encoder_dim
        self.chunk_size = chunk_size
        self.eoc_idx = eoc_idx

        self.input_embedding = rf.Embedding(vocab_dim, model_dim)
        self.pos_enc = functools.partial(rf.sinusoidal_positional_encoding, feat_dim=model_dim)
        self.input_embedding_scale = model_dim.dimension**0.5
        self.dropout = dropout

        self.layers = rf.Sequential(
            ChunkwiseDecoderLayer(
                model_dim, encoder_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout
            )
            for _ in range(num_layers)
        )
        self.final_ln = rf.LayerNorm(model_dim)
        self.logits = rf.Linear(model_dim, vocab_dim)

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        state = rf.State({k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})
        state.pos = rf.zeros((), dtype="int32", device="cpu")
        return state

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """Precompute per-layer cross-attention keys/values."""
        return rf.State(
            {k: layer.cross_att.transform_encoder(encoder, axis=axis) for k, layer in self.layers.items()}
        )

    def __call__(
        self,
        source: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
        encoder_kv: rf.State,
        enc_spatial_dim: Dim,
        query_chunk_idx: Tensor,
        key_chunk_idx: Tensor,
    ) -> Tuple[Tensor, rf.State]:
        new_state = rf.State()
        x = self.input_embedding(source) * self.input_embedding_scale
        x = x + self.pos_enc(spatial_dim=spatial_dim, offset=state.pos)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        new_state.pos = state.pos + (1 if spatial_dim == single_step_dim else spatial_dim.get_size_tensor())

        for name, layer in self.layers.items():
            keys, values = encoder_kv[name]
            x, new_state[name] = layer(
                x,
                spatial_dim=spatial_dim,
                self_att_state=state[name],
                keys=keys,
                values=values,
                enc_spatial_dim=enc_spatial_dim,
                query_chunk_idx=query_chunk_idx,
                key_chunk_idx=key_chunk_idx,
            )
        x = self.final_ln(x)
        return self.logits(x), new_state


class _FeedForward(rf.Module):
    def __init__(self, model_dim: Dim, ff_dim: Dim, *, dropout: float):
        super().__init__()
        self.lin1 = rf.Linear(model_dim, ff_dim)
        self.lin2 = rf.Linear(ff_dim, model_dim)
        self.dropout = dropout

    def __call__(self, x: Tensor) -> Tensor:
        x = self.lin1(x)
        x = rf.relu_square(x)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        return self.lin2(x)
