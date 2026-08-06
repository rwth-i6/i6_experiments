"""
Standard (full-attention) AED decoder variant for the slow-fast-rna controls.

This is NOT a streaming decoder.
It reproduces the offline AED+CTC baseline (base chunked-...-dyn-rope-ctembed, CTC-only 9.41) from scratch,
wired through ``_train_streaming_variant`` like the streaming variants,
so it shares the exact encoder / dataset pipeline / FZJ infra,
and the decoder is the only difference -- the sole variable in the comparison.

It reuses :class:`...streaming.chunkwise.ChunkwiseDecoder` (Transformer++ layers + cross-att),
but feeds it **unmasked** cross-attention indices:
``key_chunk_idx = frame index`` and ``query_chunk_idx = enc_len - 1``,
so every real encoder frame is admitted (``key <= query``) and only padding (``idx >= enc_len``) is masked --
ordinary full-context AED cross-att.
Target is the plain transcript (``target_mode="labels"``) with BOS/EOS,
label-synchronous CE (label smoothing) + the aux CTC heads.
Recog = CTC-only (``model_recog_ctc``), the 9.41 metric;
the AED decoder's own search is deferred (recog_def=None).
"""

from __future__ import annotations

from typing import Dict, Tuple, TYPE_CHECKING

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim, single_step_dim

from .base import label_smoothed_log_probs, mark_frame_error

# ChunkwiseDecoder is the decoder class used via dec_build_dict in the recipe (re-exported for convenience).
from .chunkwise import ChunkwiseDecoder  # noqa: F401

if TYPE_CHECKING:
    from i6_experiments.users.zeyer.model_interfaces import RecogDef


def standard_aed_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    labels: Tensor,
    labels_spatial_dim: Dim,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """Teacher-forced full-context AED training over the plain transcript (BOS/EOS), + aux CTC."""
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)

    # Unmasked cross-att: admit every real encoder frame, mask padding.
    # key_chunk = frame index (0..enc_padded-1), query_chunk = enc_len-1 (per seq).
    # key <= query is then true iff frame < enc_len, i.e. full context with padding masked.
    key_chunk_idx = rf.range_over_dim(enc_spatial_dim)  # [enc_spatial]
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())  # [B]
    query_chunk_idx = enc_lens - 1  # [B], broadcasts over the decoder label axis

    eos_idx = model.eoc_idx  # reuse the extra (last) vocab slot as EOS
    # input = BOS + labels, target = labels + EOS, sharing one U+1 spatial dim.
    targets_eos, (dec_spatial_dim,) = rf.pad(labels, axes=[labels_spatial_dim], padding=[(0, 1)], value=eos_idx)
    input_labels, _ = rf.pad(
        labels, axes=[labels_spatial_dim], padding=[(1, 0)], value=model.bos_idx, out_dims=[dec_spatial_dim]
    )

    encoder_kv = model.decoder.transform_encoder(enc, axis=enc_spatial_dim)
    state = model.decoder.default_initial_state(batch_dims=batch_dims)
    dec_collected_outputs = {} if model.dec_aux_logits else None
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=dec_spatial_dim,
        state=state,
        encoder_kv=encoder_kv,
        enc_spatial_dim=enc_spatial_dim,
        query_chunk_idx=query_chunk_idx,
        key_chunk_idx=key_chunk_idx,
        collected_outputs=dec_collected_outputs,
    )
    log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
    log_probs = label_smoothed_log_probs(log_probs, axis=model.target_dim_ext)  # config-gated, default off
    ce = rf.cross_entropy(
        target=targets_eos, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext
    )
    mark_frame_error(log_probs, targets=targets_eos, axis=model.target_dim_ext)
    losses: Dict[str, Tuple[Tensor, Dim]] = {"ce": (ce, dec_spatial_dim)}

    if model.dec_aux_logits:
        losses.update(
            model.dec_aux_losses(
                collected_outputs=dec_collected_outputs,
                targets=targets_eos,
                spatial_dim=dec_spatial_dim,
                axis=model.target_dim_ext,
            )
        )

    if model.enc_aux_logits:
        # aux CTC over the plain transcript.
        # The labels are already the collapsed transcript, so there is no EOC/EOS to strip.
        labels.sparse_dim = model.target_dim
        losses.update(
            model.aux_ctc_losses(
                collected_outputs=collected_outputs,
                raw_targets=labels,
                raw_spatial_dim=labels_spatial_dim,
                enc_spatial_dim=enc_spatial_dim,
            )
        )
    return losses


def standard_aed_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the plain transcript (target_mode="labels")."""
    losses = standard_aed_train_forward(
        model, data=data, data_spatial_dim=data_spatial_dim, labels=targets, labels_spatial_dim=targets_spatial_dim
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


standard_aed_training.learning_rate_control_error_measure = "ce"


def model_recog(
    *,
    model,
    data: Tensor,
    data_spatial_dim: Dim,
) -> Tuple[Tensor, Tensor, Dim, Dim]:
    """
    Full-context AED greedy recognition (beam size 1).

    Mirrors :func:`standard_aed_train_forward`'s decode regime: cross-attention admits every real
    encoder frame (``key_chunk_idx`` = frame index, ``query_chunk_idx`` = enc_len-1), and labels are
    emitted autoregressively until the reused end-of-chunk slot (= EOS) is chosen. EOS markers are
    stripped, so the returned sequence is the plain spm transcription (valid spm indices over
    ``model.target_dim_ext``, whose first entries are the spm pieces).

    :return: (seq_targets {batch,beam,out_spatial} sparse over target_dim_ext,
              seq_log_prob {batch,beam}, out_spatial_dim, beam_dim)
    """
    from returnn.frontend.tensor_array import TensorArray

    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim)

    # Full context (identical mask to training): admit every real frame, mask padding.
    key_chunk_idx = rf.range_over_dim(enc_spatial_dim)  # [enc_spatial]
    enc_lens = rf.copy_to_device(enc_spatial_dim.get_size_tensor())  # [batch]

    beam_dim = Dim(1, name="beam")
    batch_dims_ = [beam_dim] + batch_dims
    # per-seq query chunk = enc_len-1 (full context), broadcast over the beam axis.
    query_chunk_idx = (enc_lens - 1) + rf.constant(0, dims=batch_dims_, dtype=enc_lens.dtype)

    encoder_kv = model.decoder.transform_encoder(enc, axis=enc_spatial_dim)
    decoder_state = model.decoder.default_initial_state(batch_dims=batch_dims_)
    target = rf.constant(model.bos_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext)
    eos = rf.constant(model.eoc_idx, dims=batch_dims_, sparse_dim=model.target_dim_ext)
    ended = rf.constant(False, dims=batch_dims_)
    seq_log_prob = rf.constant(0.0, dims=batch_dims_)

    # Length cap: a transcript never has more spm labels than encoder frames.
    max_steps = int(rf.reduce_max(enc_lens, axis=enc_lens.dims).raw_tensor)

    i = 0
    seq_targets = TensorArray(target)
    while True:
        logits, decoder_state = model.decoder(
            target,
            spatial_dim=single_step_dim,
            state=decoder_state,
            encoder_kv=encoder_kv,
            enc_spatial_dim=enc_spatial_dim,
            query_chunk_idx=query_chunk_idx,
            key_chunk_idx=key_chunk_idx,
        )
        label_log_prob = rf.log_softmax(logits, axis=model.target_dim_ext)
        best = rf.cast(rf.reduce_argmax(label_log_prob, axis=model.target_dim_ext), "int32")
        best.sparse_dim = model.target_dim_ext
        # Carry EOS for finished seqs (their contribution is masked out below).
        best = rf.where(ended, eos, best)
        best_lp = rf.gather(label_log_prob, indices=best, axis=model.target_dim_ext)
        best_lp = rf.where(ended, 0.0, best_lp)

        seq_log_prob = seq_log_prob + best_lp
        target = best
        seq_targets = seq_targets.push_back(target)

        ended = rf.logical_or(ended, target == model.eoc_idx)
        i += 1
        if i >= max_steps or bool(rf.reduce_all(ended, axis=ended.dims).raw_tensor):
            break

    out_spatial_dim = Dim(i, name="out-spatial")
    aug_out = seq_targets.stack(axis=out_spatial_dim)  # [beam, batch, out_spatial] over target_dim_ext

    # Strip EOS markers -> plain spm label sequence (variable length per seq).
    seq_targets_out, seq_targets_spatial_dim = rf.masked_select(
        aug_out, mask=aug_out != model.eoc_idx, dims=[out_spatial_dim]
    )
    seq_targets_out.sparse_dim = model.target_dim_ext
    return seq_targets_out, seq_log_prob, seq_targets_spatial_dim, beam_dim


model_recog: RecogDef
model_recog.output_with_beam = True
model_recog.output_blank_label = None
model_recog.batch_size_dependent = False
