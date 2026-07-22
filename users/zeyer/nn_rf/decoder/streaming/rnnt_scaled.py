"""Framewise-CE monotonic RNN-T training with configurable target delay + main-loss scale/normalization.

Sweeps for the slow-fast-rna project, disentangling why full-sum beats framewise-CE:
- ``framewise_delay_frames``: shift the RNA target right by N encoder frames (extend the encoder by N
  silence frames on the right), so every label is emitted N frames after its acoustic evidence (lookahead);
  the tail flushes into the trailing silence, no labels are dropped. delay 0 == plain ``rnnt.rnnt_training``.
- ``framewise_main_loss_scale``: multiply ONLY the framewise CE main loss (aux CTC unscaled).
- ``framewise_main_loss_norm``: normalize the CE by frames T (``"frames"``, the default / same as
  ``rnnt_training``) or by labels S (``"labels"``, matching CTC + the full-sum normalization) -- tests
  whether the ~T/S stronger encoder gradient of the full-sum objective is what helps its decoder.

Same ``RnntDecoder`` model as ``rnnt`` / ``rnnt_fullsum`` (comparable); only the objective's timing/scale/norm
change, all via opt-in config keys (default = ``rnnt_training``), so existing jobs do not rehash.
"""

from __future__ import annotations

from typing import Dict, Tuple

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim

from .base import rna_targets_on_enc_spatial, label_smoothed_log_probs, mark_frame_error


def rnnt_scaled_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    rna_targets: Tensor,
    rna_targets_spatial_dim: Dim,
    delay_frames: int = 0,
) -> Tuple[Dict[str, Tuple[Tensor, Dim]], Dim]:
    """Framewise-CE forward (as ``rnnt.rnnt_train_forward``) with an optional target delay.

    The delay extends the encoder by ``delay_frames`` silence frames on the right and shifts the RNA
    target right by the same amount (prepend blanks) onto the extended axis (mirrors ``framewise``);
    the CE runs on the extended axis while the aux CTC stays on the original encoder output.

    :return: ``(losses, label_spatial_dim)`` -- ``label_spatial_dim`` (= #labels S) is returned so the
        train def can normalize the CE by labels instead of frames.
    """
    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    rna = rna_targets_on_enc_spatial(
        rna_targets, in_spatial_dim=rna_targets_spatial_dim, enc_spatial_dim=enc_spatial_dim, blank_idx=model.blank_idx
    )
    batch_dims = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)
    dec = model.decoder
    blank = model.blank_idx

    # Delay: extend the encoder by delay_frames silence frames (right) and shift the RNA target right by
    # the same amount (prepend blanks) onto the extended axis. delay == 0 -> numerically the pre-delay path.
    if delay_frames > 0:
        enc_dec, (dec_spatial_dim,) = rf.pad(enc, axes=[enc_spatial_dim], padding=[(0, delay_frames)], value=0.0)
        rna_dec, _ = rf.pad(
            rna, axes=[enc_spatial_dim], padding=[(delay_frames, 0)], value=blank, out_dims=[dec_spatial_dim]
        )
    else:
        enc_dec, dec_spatial_dim, rna_dec = enc, enc_spatial_dim, rna

    valid_frame = rf.range_over_dim(dec_spatial_dim) < rf.copy_to_device(dec_spatial_dim.get_size_tensor())
    is_label = rf.logical_and(rna_dec != blank, valid_frame)
    is_label_i = rf.cast(is_label, "int32")
    n_t = rf.cumsum(is_label_i, spatial_dim=dec_spatial_dim) - is_label_i  # [B, dec_spatial]: #labels before t

    y, label_spatial_dim = rf.masked_select(rna_dec, mask=is_label, dims=[dec_spatial_dim])
    y.sparse_dim = model.target_dim_ext

    pred_in, (label_ext_dim,) = rf.pad(y, axes=[label_spatial_dim], padding=[(1, 0)], value=model.bos_idx)
    pred_in.sparse_dim = model.target_dim_ext
    pred, _ = dec.pred_forward(
        pred_in, spatial_dim=label_ext_dim, state=dec.pred_initial_state(batch_dims=batch_dims)
    )
    pred_per_frame = rf.gather(pred, indices=n_t, axis=label_ext_dim)

    logits = dec.joiner(enc_dec, pred_per_frame)
    log_probs = rf.log_softmax(logits, axis=model.target_dim_ext)
    log_probs = label_smoothed_log_probs(log_probs, axis=model.target_dim_ext)  # config-gated, default off
    ce = rf.cross_entropy(target=rna_dec, estimated=log_probs, estimated_type="log-probs", axis=model.target_dim_ext)
    mark_frame_error(log_probs, targets=rna_dec, axis=model.target_dim_ext)
    losses: Dict[str, Tuple[Tensor, Dim]] = {"ce": (ce, dec_spatial_dim)}

    # Aux CTC on the blank-removed labels (delay-invariant), over the ORIGINAL encoder output.
    if model.enc_aux_logits:
        raw_targets, raw_spatial_dim = rf.masked_select(rna, mask=rna != blank, dims=[enc_spatial_dim])
        raw_targets.sparse_dim = model.target_dim
        losses.update(
            model.aux_ctc_losses(
                collected_outputs=collected_outputs,
                raw_targets=raw_targets,
                raw_spatial_dim=raw_spatial_dim,
                enc_spatial_dim=enc_spatial_dim,
            )
        )
    return losses, label_spatial_dim


def rnnt_scaled_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: framewise CE (RNA target) with config-driven delay + main-loss scale + norm (frames/labels)."""
    from returnn.config import get_global_config

    cfg = get_global_config()
    delay = cfg.int("framewise_delay_frames", 0)
    scale = cfg.float("framewise_main_loss_scale", 1.0)
    norm = cfg.value("framewise_main_loss_norm", "frames")
    assert norm in ("frames", "labels"), norm

    losses, label_spatial_dim = rnnt_scaled_train_forward(
        model,
        data=data,
        data_spatial_dim=data_spatial_dim,
        rna_targets=targets,
        rna_targets_spatial_dim=targets_spatial_dim,
        delay_frames=delay,
    )
    for name, (loss, norm_dim) in losses.items():
        if name == "ce":
            inv_dim = label_spatial_dim if norm == "labels" else norm_dim
            loss.mark_as_loss(name, scale=scale, custom_inv_norm_factor=inv_dim.get_size_tensor(), use_normalized_loss=True)
        else:
            loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


rnnt_scaled_training.learning_rate_control_error_measure = "ce"
