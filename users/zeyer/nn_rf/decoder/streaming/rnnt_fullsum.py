"""
Full-sum (marginalized) monotonic RNN-T training for the standard-monotonic ``RnntDecoder`` (version >= 3).

Companion to :mod:`rnnt`, which holds the model + the framewise-CE train def.
The SAME v3 model trains with either framewise-CE (``rnnt.rnnt_training``, our fixed RNA alignment)
or the marginalized loss here,
switching only the objective.

Loss = ``i6_native_ops.monotonic_rnnt.monotonic_rnnt_loss`` (monotonic / RNA topology, softmax internal).
It also accepts an optional ``alignment`` + ``max_shift_from_alignment``:
shift 0 = Viterbi (== framewise on that alignment),
larger = marginalization restricted to a band,
``alignment=None`` = full marginalization.

Efficiency: the additive joiner pre-activation ``enc_proj[t] + pred[u]`` is packed to the valid
(t, u) cells before ``relu`` + the vocab projection,
by gathering the enc and pred rows separately into the packed layout and adding
(the full ``[batch, enc_spatial, s1, model_dim]`` broadcast is never materialized),
so the expensive ``V``-projection runs only on ``sum_b T_b*(S_b+1)`` cells
-- the packed layout the kernel expects (b-major, t outer, u inner, no padding).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim


def rnnt_fullsum_train_forward(
    model,
    *,
    data: Tensor,
    data_spatial_dim: Dim,
    labels: Tensor,
    labels_spatial_dim: Dim,
    max_shift_from_alignment: int = 0,
    alignment: Optional[Tensor] = None,
    alignment_spatial_dim: Optional[Dim] = None,
) -> Dict[str, Tuple[Tensor, Dim]]:
    """Marginalized monotonic RNN-T forward.

    ``labels`` is the plain transcription (``target_mode="labels"``),
    NOT a per-frame alignment.
    Returns ``{loss_name: (loss, norm_dim)}``:
    the monotonic-RNNT cost + the aux CTC losses.
    """
    import os
    import sys

    # i6_native_ops JIT-builds/verifies its CUDA kernel via ninja, which must be on PATH;
    # the env's bin/ (where the installed ninja lives) is not always on the job's PATH.
    _bindir = os.path.dirname(sys.executable)
    if _bindir not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = _bindir + os.pathsep + os.environ.get("PATH", "")
    from i6_native_ops import monotonic_rnnt

    dec = model.decoder
    assert getattr(dec, "version", 2) >= 3, "full-sum needs the standard monotonic RNN-T (RnntDecoder version>=3)"

    collected_outputs = {} if model.enc_aux_logits else None
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    (batch_dim,) = data.remaining_dims((data_spatial_dim, data.feature_dim) if data.feature_dim else data_spatial_dim)

    # Prediction net over [BOS, y_1..y_S] -> g_0..g_S (length S+1); v3 pred net is label-history only.
    y = labels
    pred_in, (s1_dim,) = rf.pad(labels, axes=[labels_spatial_dim], padding=[(1, 0)], value=model.bos_idx)
    pred_in.sparse_dim = model.target_dim_ext
    pred, _ = dec.pred_forward(pred_in, spatial_dim=s1_dim, state=dec.pred_initial_state(batch_dims=[batch_dim]))

    # Additive joiner pre-activation over the (t, u) grid, packed WITHOUT the dense broadcast.
    # Gather the enc and pred rows separately (over the merged batch+time / batch+label axes) and add,
    # so the full {batch, enc_spatial, s1, model_dim} tensor is never materialized.
    # Peak model_dim tensor is only [sum_b T_b*(S_b+1), model_dim], so the batch can grow.
    enc_proj = dec.joiner_enc_proj(enc)  # {batch, enc_spatial, model_dim}

    # Valid-cell mask: t < T_b  and  u < S_b + 1.
    t_len = rf.copy_to_device(enc_spatial_dim.get_size_tensor())
    s_len = rf.copy_to_device(labels_spatial_dim.get_size_tensor())
    t_valid = rf.range_over_dim(enc_spatial_dim) < t_len
    u_valid = rf.range_over_dim(s1_dim) < (s_len + 1)
    mask = rf.logical_and(t_valid, u_valid)

    # Flat row indices into the merged (batch, enc_spatial) / (batch, s1) axes (padded stride).
    t_max = int(enc_spatial_dim.get_dim_value())
    u_max = int(s1_dim.get_dim_value())
    flat_t = rf.range_over_dim(batch_dim) * t_max + rf.range_over_dim(enc_spatial_dim)
    flat_u = rf.range_over_dim(batch_dim) * u_max + rf.range_over_dim(s1_dim)

    # Pack the flat indices onto one shared packed_dim (b-major, t outer, u inner).
    packed_t, packed_dim = rf.masked_select(flat_t, mask=mask, dims=[batch_dim, enc_spatial_dim, s1_dim])
    packed_u, _ = rf.masked_select(flat_u, mask=mask, dims=[batch_dim, enc_spatial_dim, s1_dim], out_dim=packed_dim)

    # Gather the packed rows and add, THEN relu + vocab projection.
    enc_merged, bt_dim = rf.merge_dims(enc_proj, dims=[batch_dim, enc_spatial_dim])
    pred_merged, bu_dim = rf.merge_dims(pred, dims=[batch_dim, s1_dim])
    packed_pre = rf.gather(enc_merged, indices=packed_t, axis=bt_dim) + rf.gather(
        pred_merged, indices=packed_u, axis=bu_dim
    )
    acts = dec.logits(rf.relu(packed_pre))  # [packed, V]

    acts_raw = acts.copy_compatible_to_dims_raw([packed_dim, model.target_dim_ext]).float()
    labels_raw = y.copy_compatible_to_dims_raw([batch_dim, labels_spatial_dim]).int()
    input_lengths = t_len.copy_compatible_to_dims_raw([batch_dim]).int()
    label_lengths = s_len.copy_compatible_to_dims_raw([batch_dim]).int()
    align_raw = None
    if alignment is not None:
        align_raw = alignment.copy_compatible_to_dims_raw([batch_dim, alignment_spatial_dim]).int()

    costs = monotonic_rnnt.monotonic_rnnt_loss(
        acts=acts_raw,
        labels=labels_raw,
        input_lengths=input_lengths,
        label_lengths=label_lengths,
        alignment=align_raw,
        max_shift_from_alignment=max_shift_from_alignment,
        blank_label=model.blank_idx,
    )  # [B] neg-log-prob
    cost = rf.convert_to_tensor(costs, dims=[batch_dim], name="mono_rnnt_cost")

    losses: Dict[str, Tuple[Tensor, Dim]] = {"mono_rnnt": (cost, labels_spatial_dim)}
    if model.enc_aux_logits:
        losses.update(
            model.aux_ctc_losses(
                collected_outputs=collected_outputs,
                raw_targets=y,
                raw_spatial_dim=labels_spatial_dim,
                enc_spatial_dim=enc_spatial_dim,
            )
        )
    return losses


def rnnt_fullsum_training(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """TrainDef: ``targets`` is the plain transcription (``target_mode="labels"``)."""
    losses = rnnt_fullsum_train_forward(
        model, data=data, data_spatial_dim=data_spatial_dim, labels=targets, labels_spatial_dim=targets_spatial_dim
    )
    for name, (loss, norm_dim) in losses.items():
        loss.mark_as_loss(name, custom_inv_norm_factor=norm_dim.get_size_tensor(), use_normalized_loss=True)


rnnt_fullsum_training.learning_rate_control_error_measure = "mono_rnnt"
