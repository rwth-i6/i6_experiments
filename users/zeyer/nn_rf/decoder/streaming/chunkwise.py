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

from typing import Optional, Any, Dict, Sequence, Tuple, List, TYPE_CHECKING

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim
from returnn.frontend.decoder.transformer import FeedForwardGated

from .cross_attn import ChunkMaskedCrossAttention
from .base import encoder_frame_chunk_idx, label_smoothed_log_probs, mark_frame_error

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


class ChunkwiseDecoderLayer(rf.Module):
    """Transformer++ (Llama-style) block as in the AED baseline decoder:
    RoPE causal self-att (no bias), RMSNorm, gated FF,
    plus the chunk-masked cross-attention to the encoder.
    """

    def __init__(
        self,
        model_dim: Dim,
        encoder_dim: Dim,
        ff_dim: Optional[Dim],
        *,
        num_heads: int,
        dropout: float,
        att_dropout: float,
    ):
        super().__init__()
        self.self_att_ln = rf.RMSNorm(model_dim)
        self.self_att = rf.RotaryPosCausalSelfAttention(
            model_dim,
            proj_dim=model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=num_heads,
            with_bias=False,
            att_dropout=att_dropout,
        )
        self.cross_att_ln = rf.RMSNorm(model_dim)
        self.cross_att = ChunkMaskedCrossAttention(
            encoder_dim,
            model_dim,
            key_dim_total=model_dim,
            value_dim_total=model_dim,
            num_heads=num_heads,
            att_dropout=att_dropout,
        )
        self.ff_ln = rf.RMSNorm(model_dim)
        self.ff = FeedForwardGated(model_dim, ff_dim=ff_dim, dropout=dropout)
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
        ff_dim: Optional[int] = None,  # None -> FeedForwardGated default (Llama-style ~8/3 * model_dim)
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        att_dropout: float = 0.1,
        version: int = 1,
    ):
        super().__init__()
        # v1 = the pre-Transformer++ decoder (LayerNorm + abs sin pos-enc + non-gated FF); that code is gone.
        # v2 = RMSNorm + RoPE causal self-att + gated FF.
        # rf.build_dict hashes the dict, not the module source,
        # so this explicit version is what forces a new sis hash for the rewrite.
        assert version >= 2, "ChunkwiseDecoder v1 (pre-Transformer++) is removed; build with version=2"
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
        self.input_embedding_scale = model_dim.dimension**0.5
        self.dropout = dropout

        self.layers = rf.Sequential(
            ChunkwiseDecoderLayer(
                model_dim, encoder_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout
            )
            for _ in range(num_layers)
        )
        self.final_ln = rf.RMSNorm(model_dim)
        self.logits = rf.Linear(model_dim, vocab_dim)

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        return rf.State({k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})

    def transform_encoder(self, encoder: Tensor, *, axis: Dim) -> rf.State:
        """Precompute per-layer cross-attention keys/values."""
        return rf.State({k: layer.cross_att.transform_encoder(encoder, axis=axis) for k, layer in self.layers.items()})

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
        collected_outputs: Optional[Dict[str, Tensor]] = None,
    ) -> Tuple[Tensor, rf.State]:
        new_state = rf.State()
        x = self.input_embedding(source) * self.input_embedding_scale
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)

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
            if collected_outputs is not None:
                collected_outputs[name] = x
        x = self.final_ln(x)
        return self.logits(x), new_state


def chunkwise_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    aug_targets: Tensor,
    aug_targets_spatial_dim: Dim,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """
    Teacher-forced forward for chunk-synchronous training.

    ``aug_targets`` is the EOC-augmented label sequence (the single supervision
    stream). Everything else is derived in-graph:
    the per-position chunk index is the exclusive prefix count of EOC tokens, and
    the raw spm labels (for the aux CTC head) are ``aug_targets`` with EOC removed.

    :return: dict ``name -> (loss, inv_norm_spatial_dim)``.
    """
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    key_chunk_idx = encoder_frame_chunk_idx(enc_spatial_dim, model.chunk_size)

    # Per-position chunk index = number of EOC tokens strictly before each position.
    is_eoc = rf.cast(aug_targets == model.eoc_idx, "int32")
    pos_chunk_idx = rf.cumsum(is_eoc, spatial_dim=aug_targets_spatial_dim) - is_eoc

    # Teacher forcing: decoder input is the right-shifted target, seeded with BOS.
    input_labels = rf.shift_right(aug_targets, axis=aug_targets_spatial_dim, pad_value=model.bos_idx)

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    encoder_kv = model.decoder.transform_encoder(enc, axis=enc_spatial_dim)
    state = model.decoder.default_initial_state(batch_dims=batch_dims)
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=aug_targets_spatial_dim,
        state=state,
        encoder_kv=encoder_kv,
        enc_spatial_dim=enc_spatial_dim,
        query_chunk_idx=pos_chunk_idx,
        key_chunk_idx=key_chunk_idx,
    )
    log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
    log_probs = label_smoothed_log_probs(log_probs, axis=model.target_dim_ext)  # config-gated, default off
    ce = rf.cross_entropy(
        target=aug_targets, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext
    )
    mark_frame_error(log_probs, targets=aug_targets, axis=model.target_dim_ext)
    losses: Dict[str, Tuple[Tensor, Dim]] = {"ce": (ce, aug_targets_spatial_dim)}

    # Aux CTC on the raw spm labels (aug_targets with EOC removed), over the final encoder output.
    if model.enc_aux_logits:
        raw_targets, raw_spatial_dim = rf.masked_select(
            aug_targets, mask=aug_targets != model.eoc_idx, dims=[aug_targets_spatial_dim]
        )
        raw_targets.sparse_dim = model.target_dim
        losses.update(
            model.aux_ctc_losses(
                collected_outputs=collected_outputs,
                raw_targets=raw_targets,
                raw_spatial_dim=raw_spatial_dim,
                enc_spatial_dim=enc_spatial_dim,
            )
        )
    return losses


def chunkwise_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the EOC-augmented label sequence (the default target)."""
    losses = chunkwise_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        aug_targets=targets,
        aug_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


chunkwise_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Chunk-synchronous greedy recognition (beam size 1).

    For each encoder chunk, emit labels autoregressively until the EOC marker, then
    advance to the next chunk; stop once all chunks are consumed (or a per-chunk /
    global step cap is hit). EOC markers are removed from the returned sequence, so
    it is the plain spm transcription -- the labels are valid spm indices, rendered
    via ``model.target_dim_ext``'s vocab (whose first entries are the spm pieces).

    :return: (seq_targets {batch,beam,out_spatial} sparse over target_dim_ext,
              seq_log_prob {batch,beam}, out_spatial_dim, beam_dim)
    """
    from returnn.config import get_global_config
    from returnn.frontend.tensor_array import TensorArray

    config = get_global_config(return_empty_if_none=True)
    max_labels_per_chunk = config.int("max_labels_per_chunk", 0) or 20

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    key_chunk_idx = encoder_frame_chunk_idx(enc_spatial_dim, model.chunk_size)  # [enc_spatial]
    # number of chunks per seq = ceil(enc_len / chunk_size). Derive from the per-seq
    # encoder lengths (key_chunk_idx itself lacks the batch axis, so can't masked-reduce it).
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())  # [batch]
    num_chunks = (enc_lens - 1) // model.chunk_size + 1  # [batch]

    beam_dim = Dim(1, name="beam")
    batch_dims_ = [beam_dim] + batch_dims
    encoder_kv = model.decoder.transform_encoder(enc, axis=enc_spatial_dim)
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext)
    eoc = rf.constant(model.eoc_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext)
    cur_chunk = rf.constant(0, dims=batch_dims_, dtype="int32")
    labels_in_chunk = rf.constant(0, dims=batch_dims_, dtype="int32")
    ended = rf.constant(False, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    # Global step cap: each chunk emits at most max_labels_per_chunk labels + 1 EOC.
    max_steps = int(rf.reduce_max(num_chunks, axis=num_chunks.dims).raw_tensor) * (max_labels_per_chunk + 1)

    i = 0
    seq_targets = TensorArray(target)
    while True:
        logits, decoder_state = model.decoder(
            target,
            spatial_dim=single_step_dim,
            state=decoder_state,
            encoder_kv=encoder_kv,
            enc_spatial_dim=enc_spatial_dim,
            query_chunk_idx=cur_chunk,
            key_chunk_idx=key_chunk_idx,
        )
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim_ext)
        best = rf.cast(rf.reduce_argmax(label_log_prob, axis=model.target_dim_ext), "int32")
        best.sparse_dim = model.target_dim_ext
        # Force EOC if this chunk hit its label budget, or carry EOC for finished seqs.
        force_eoc = rf.logical_or(labels_in_chunk >= max_labels_per_chunk, ended)
        best = rf.where(force_eoc, eoc, best)
        best_lp = rf.gather(label_log_prob, indices=best, axis=model.target_dim_ext)
        best_lp = rf.where(ended, 0.0, best_lp)

        seq_log_prob = seq_log_prob + best_lp
        target = best
        seq_targets = seq_targets.push_back(target)

        is_eoc = target == model.eoc_idx
        cur_chunk = cur_chunk + rf.cast(is_eoc, "int32")
        labels_in_chunk = rf.where(is_eoc, rf.constant(0, dims=batch_dims_, dtype="int32"), labels_in_chunk + 1)
        ended = rf.logical_or(ended, cur_chunk >= num_chunks)
        i += 1
        if i >= max_steps or bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break

    out_spatial_dim = Dim(i, name="out-spatial")
    aug_out = seq_targets.stack(axis=out_spatial_dim)  # [beam, batch, out_spatial] over target_dim_ext

    # Strip EOC markers -> plain spm label sequence (variable length per seq).
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(
        aug_out, mask=aug_out != model.eoc_idx, dims=[out_spatial_dim]
    )
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False
