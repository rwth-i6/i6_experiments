"""Encoder-only alignment-quality probes.

Two train defs that supervise ONLY the encoder aux CTC heads (no streaming-decoder loss),
to compare the fixed CTC forced-alignment against standard CTC on the same encoder:

- :func:`aux_framewise_ce_training`: per-frame CE on each aux head against the raw CTC
  alignment (``target_mode="ctc_frame"``) -- commits the encoder to the fixed alignment.
- :func:`aux_ctc_only_training`: standard CTC on each aux head against the transcript
  (``target_mode="labels"``) -- marginalizes over alignments.

The model's decoder is still built (``streaming_model_def`` requires one) but never called,
for architectural parity with the RNN-T-small pair; recog uses ``model_recog_ctc`` on the
top aux head. Reading: A close to B means the CTC alignment is essentially CTC-optimal;
A much worse than B means the fixed alignment is a lossy per-frame target. Either way this
says nothing about how good the alignment is as an RNA target for the transducer variants.
"""

from __future__ import annotations

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

from .base import StreamingModel, rna_targets_on_enc_spatial


def aux_framewise_ce_training(
    *, model: StreamingModel, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim
):
    """TrainDef: framewise CE on every encoder aux head against the raw CTC alignment (``ctc_frame``)."""
    assert model.enc_aux_logits, "alignment probe needs aux heads (set aux_loss_layers)"
    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    # Re-tag the per-frame CTC alignment onto the encoder length (pad blank / cut blank padding),
    # so the encoder chunking may differ from the dataset's fixed pad-to-chunk-multiple.
    frame_targets = rna_targets_on_enc_spatial(
        targets, in_spatial_dim=targets_spatial_dim, enc_spatial_dim=enc_spatial_dim, blank_idx=model.blank_idx
    )
    frame_targets.sparse_dim = model.wb_target_dim
    for layer_idx in model.enc_aux_logits:
        aux_logits = model.aux_logits_from_collected_outputs(layer_idx, collected_outputs)
        log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        ce = rf.cross_entropy(
            target=frame_targets, estimated=log_probs, estimated_type="log-probs", axis=model.wb_target_dim
        )
        ce.mark_as_loss(
            f"ce_{layer_idx}", custom_inv_norm_factor=enc_spatial_dim.get_size_tensor(), use_normalized_loss=True
        )


aux_framewise_ce_training.learning_rate_control_error_measure = "ce"


def aux_ctc_only_training(
    *, model: StreamingModel, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim
):
    """TrainDef: standard CTC on every encoder aux head against the transcript (``labels``)."""
    assert model.enc_aux_logits, "alignment probe needs aux heads (set aux_loss_layers)"
    collected_outputs = {}
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    losses = model.aux_ctc_losses(
        collected_outputs=collected_outputs,
        raw_targets=targets,
        raw_spatial_dim=targets_spatial_dim,
        enc_spatial_dim=enc_spatial_dim,
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


aux_ctc_only_training.learning_rate_control_error_measure = "ctc"
