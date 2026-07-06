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

from typing import Dict, Tuple

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

from .base import label_smoothed_log_probs

# ChunkwiseDecoder is the decoder class used via dec_build_dict in the recipe (re-exported for convenience).
from .chunkwise import ChunkwiseDecoder  # noqa: F401


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
