"""
Frame-synchronous (RNA) fast-only streaming decoder (slow-fast-rna project).

A single Transformer stack runs at encoder-frame rate: per encoder frame it emits one
label or blank (RNA topology -- the non-blank labels, no repeat collapse, are the
transcription). Each position consumes the current encoder frame ``h_t`` (added in) plus
the previous frame's symbol (teacher-forced in training, fed back at recog); causal
self-attention over frames carries the label/acoustic history. No cross-attention: the
chunked encoder already supplies streaming-causal acoustic context in ``h_t``.

This is the fast-only baseline and the *fast stack* that ``ext_transducer`` extends with a
slow label-rate stack. Target: the per-frame RNA alignment (``segmentation.rna_frame_targets``)
padded to the encoder's chunk-multiple length, so it lines up with the encoder output
frame-for-frame.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple, TYPE_CHECKING
import functools

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


class FramewiseDecoderLayer(rf.Module):
    def __init__(self, model_dim: Dim, ff_dim: Dim, *, num_heads: int, dropout: float, att_dropout: float):
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
        self.ff_ln = rf.LayerNorm(model_dim)
        self.ff = _FeedForward(model_dim, ff_dim, dropout=dropout)
        self.dropout = dropout

    def __call__(self, x: Tensor, *, spatial_dim: Dim, self_att_state: rf.State) -> Tuple[Tensor, rf.State]:
        h, new_state = self.self_att(self.self_att_ln(x), spatial_dim, state=self_att_state)
        x = x + rf.dropout(h, self.dropout, axis=h.feature_dim)
        x = x + self.ff(self.ff_ln(x))
        return x, new_state


class FramewiseDecoder(rf.Module):
    """Frame-synchronous RNA decoder: causal self-attn over frames + per-frame encoder input."""

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
        self.blank_idx = eoc_idx  # the extra (last) vocab symbol is the RNA blank here

        self.input_embedding = rf.Embedding(vocab_dim, model_dim)
        self.enc_proj = rf.Linear(encoder_dim, model_dim, with_bias=False)
        self.pos_enc = functools.partial(rf.sinusoidal_positional_encoding, feat_dim=model_dim)
        self.input_embedding_scale = model_dim.dimension**0.5
        self.dropout = dropout

        self.layers = rf.Sequential(
            FramewiseDecoderLayer(model_dim, ff_dim, num_heads=num_heads, dropout=dropout, att_dropout=att_dropout)
            for _ in range(num_layers)
        )
        self.final_ln = rf.LayerNorm(model_dim)
        self.logits = rf.Linear(model_dim, vocab_dim)

    def default_initial_state(self, *, batch_dims: Sequence[Dim]) -> rf.State:
        state = rf.State({k: v.self_att.default_initial_state(batch_dims=batch_dims) for k, v in self.layers.items()})
        state.pos = rf.zeros((), dtype="int32", device="cpu")
        return state

    def __call__(
        self,
        source: Tensor,
        enc_frame: Tensor,
        *,
        spatial_dim: Dim,
        state: rf.State,
    ) -> Tuple[Tensor, rf.State]:
        """
        :param source: previous-frame symbol(s), sparse over ``vocab_dim``, on ``spatial_dim``.
        :param enc_frame: encoder output aligned to ``source`` (same ``spatial_dim``).
        :param spatial_dim: the frame axis (``enc_spatial_dim`` in training, ``single_step_dim`` at recog).
        :param state: self-attn state + position counter.
        """
        new_state = rf.State()
        x = self.input_embedding(source) * self.input_embedding_scale
        x = x + self.enc_proj(enc_frame)
        x = x + self.pos_enc(spatial_dim=spatial_dim, offset=state.pos)
        x = rf.dropout(x, self.dropout, axis=x.feature_dim)
        new_state.pos = state.pos + (1 if spatial_dim == single_step_dim else spatial_dim.get_size_tensor())

        for name, layer in self.layers.items():
            x, new_state[name] = layer(x, spatial_dim=spatial_dim, self_att_state=state[name])
        x = self.final_ln(x)
        return self.logits(x), new_state


def framewise_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    rna_targets: Tensor,
    rna_targets_spatial_dim: Dim,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """
    Teacher-forced frame-synchronous RNA training.

    ``rna_targets`` is the per-frame RNA alignment (label or blank per frame), padded by
    the dataset to the encoder's chunk-multiple length, so it matches ``enc_spatial_dim``
    frame-for-frame (we just re-tag the dim). Loss = framewise CE; optional aux CTC on the
    blank-removed labels (the transcription).

    :return: dict ``name -> (loss, inv_norm_spatial_dim)``.
    """
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    # Dataset padded the RNA target to ceil(T_align/chunk)*chunk == T_enc; relabel the dim
    # onto enc_spatial_dim (same per-seq sizes; replace_dim is a tag swap, not size-checked).
    rna, _ = rf.replace_dim(rna_targets, in_dim=rna_targets_spatial_dim, out_dim=enc_spatial_dim)

    # Teacher forcing: decoder input at frame t is the previous frame's symbol (BOS at t=0).
    input_labels = rf.shift_right(rna, axis=enc_spatial_dim, pad_value=model.bos_idx)

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    state = model.decoder.default_initial_state(batch_dims=batch_dims)
    logits, _ = model.decoder(input_labels, enc, spatial_dim=enc_spatial_dim, state=state)
    log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
    ce = rf.cross_entropy(target=rna, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext)
    losses: Dict[str, Tuple[Tensor, Dim]] = {"ce": (ce, enc_spatial_dim)}

    # Aux CTC on the blank-removed labels (== transcription), over the final encoder output.
    if model.enc_aux_logits:
        raw_targets, raw_spatial_dim = rf.masked_select(rna, mask=rna != model.blank_idx, dims=[enc_spatial_dim])
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


def framewise_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the per-frame RNA alignment (the default target)."""
    losses = framewise_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


framewise_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Frame-synchronous greedy recognition (beam size 1).

    Runs exactly ``T_enc`` steps (one per encoder frame); each step feeds the previous
    emitted symbol + the current encoder frame, argmaxes over labels+blank, and feeds it
    back. Blanks are removed from the output, giving the plain spm transcription (rendered
    via ``target_dim_ext``'s vocab; the blank == the last/extra symbol).

    :return: (seq_targets {batch,beam,out_spatial} sparse over target_dim_ext,
              seq_log_prob {batch,beam}, out_spatial_dim, beam_dim)
    """
    from returnn.frontend.tensor_array import TensorArray

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())  # [batch]
    T_max = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor)

    beam_dim = Dim(1, name="beam")
    batch_dims_ = [beam_dim] + batch_dims
    state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    prev = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext, dtype="int32")
    blank = rf.constant(model.blank_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext, dtype="int32")
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    seq = TensorArray(prev)
    for t in range(T_max):
        t_t = rf.constant(t, dims=batch_dims, dtype="int32")  # [batch]
        valid = t_t < enc_lens  # [batch]: frame t is a real encoder frame for this seq
        idx = rf.where(valid, t_t, enc_lens - 1)  # clip out-of-range (batch padding) frames
        enc_t = rf.gather(enc, indices=idx, axis=enc_spatial_dim)  # [batch, enc_dim]
        logits, state = model.decoder(prev, enc_t, spatial_dim=single_step_dim, state=state)
        log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
        sym = rf.cast(rf.reduce_argmax(log_probs, axis=model.target_dim_ext), "int32")
        sym.sparse_dim = model.target_dim_ext
        sym = rf.where(valid, sym, blank)  # batch-padding frames emit blank (dropped later)
        lp_sym = rf.gather(log_probs, indices=sym, axis=model.target_dim_ext)
        seq_log_prob = seq_log_prob + rf.where(valid, lp_sym, 0.0)
        prev = sym
        seq = seq.push_back(sym)

    out_spatial_dim = Dim(T_max, name="out-spatial")
    aligned = seq.stack(axis=out_spatial_dim)  # [beam, batch, T_max] over target_dim_ext

    # Strip blanks -> plain spm label sequence (variable length per seq).
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(
        aligned, mask=aligned != model.blank_idx, dims=[out_spatial_dim]
    )
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False


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
