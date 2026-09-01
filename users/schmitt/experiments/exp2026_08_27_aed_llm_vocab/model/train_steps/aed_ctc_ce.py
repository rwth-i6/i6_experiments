"""
Train step for the AED model: label-wise CE on the decoder + aux CTC on the encoder.

Ported from :func:`i6_experiments.users.zeyer.experiments.exp2024_04_23_baselines.aed.aed_training`,
behaviour-identical at the time of the port (2026-08-27).

Packed-tensor note: this step is packing-agnostic on purpose, and that is not an accident of the
port -- every packed Loquacious run in ``exp2026_05_23_returnn.py`` uses this very train step
unmodified. Two details are what make that work:

- ``rf.ctc_loss`` dispatches to RETURNN's packed native fast-BW op when the inputs are packed.
  The alternative (unpack to a padded [B,T,V] intermediate, then ``aten._ctc_loss``) is untraceable
  under fake tensors (DynamicOutputShapeException), so it could not run compiled/captured at all --
  which is exactly why the packed native op exists.
- ``rf.pack_padded`` on the decoder side is a no-op-ish rearrangement when the targets are already
  packed, so the CE path needs no special case either.

``ctc_use_native_op`` is kept as the diagnostic switch to force the generic (aten) CTC; it is only
useful for A/B debugging and will disable compile/capture.
"""

from __future__ import annotations

import functools

import returnn.frontend as rf
from returnn.tensor import Dim

from i6_experiments.users.zeyer.model_interfaces import TrainDef

from ..definitions.aed import Model, log_probs_with_eos_separated

__all__ = ["train_def"]


def train_def(*, model: Model, data: rf.Tensor, data_spatial_dim: Dim, targets: rf.Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config
    from returnn.util.collect_outputs_dict import CollectOutputsDict

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers") or ()
    aux_loss_scales = config.typed_value("aux_loss_scales", [1.0] * len(aux_loss_layers))
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    dec_aux_loss_layers = config.typed_value("dec_aux_loss_layers") or ()
    dec_aux_loss_scales = config.typed_value("dec_aux_loss_scales", [1.0] * len(dec_aux_loss_layers))
    use_normalized_loss = config.typed_value("use_normalized_loss", True)
    if isinstance(use_normalized_loss, bool):
        use_normalized_loss = "frames" if use_normalized_loss else "none"
    assert isinstance(use_normalized_loss, str) and use_normalized_loss in ("none", "frames", "seqs")
    label_smoothing = config.float("label_smoothing", 0.1)
    aux_ctc_label_smoothing = config.float("aux_ctc_label_smoothing", 0.0)
    text_augment = config.typed_value("text_augment", None)

    ctc_loss = rf.ctc_loss
    if aux_ctc_label_smoothing:
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        ctc_loss = ctc_loss_fixed_grad
    ctc_use_native_op = config.typed_value("ctc_use_native_op", None)
    if ctc_use_native_op is not None:
        # diagnostic switch, e.g. force the generic (aten) CTC under packed training
        ctc_loss = functools.partial(ctc_loss, use_native_op=ctc_use_native_op)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        ctc_targets, (ctc_targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )
    else:
        ctc_targets, ctc_targets_spatial_dim = targets, targets_spatial_dim

    collected_outputs = CollectOutputsDict(allowed_key_patterns=[str(layer_idx - 1) for layer_idx in aux_loss_layers])
    enc, enc_spatial_dim = model.encode(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    for i, layer_idx in enumerate(aux_loss_layers):
        if layer_idx > len(model.encoder.layers):
            continue
        linear = getattr(model, f"enc_aux_logits_{layer_idx}")
        aux_logits = linear(collected_outputs[str(layer_idx - 1)])
        aux_ctc_log_probs = rf.log_softmax(aux_logits, axis=model.wb_target_dim)
        if aux_ctc_label_smoothing:
            aux_ctc_log_probs = rf.label_smoothed_log_prob_gradient(
                aux_ctc_log_probs, smoothing=aux_ctc_label_smoothing, axis=model.wb_target_dim
            )
        aux_loss = ctc_loss(
            logits=aux_ctc_log_probs,
            logits_normalized=True,
            targets=ctc_targets,
            input_spatial_dim=enc_spatial_dim,
            targets_spatial_dim=ctc_targets_spatial_dim,
            blank_index=model.blank_idx,
        )
        if use_normalized_loss in ("none", "frames"):
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i],
                custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(device=targets.device),
                use_normalized_loss={"none": False, "frames": True}[use_normalized_loss],
            )
        elif use_normalized_loss == "seqs":
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=0,
                custom_inv_norm_factor=ctc_targets_spatial_dim.get_size_tensor(device=targets.device),
            )
            aux_loss.mark_as_loss(f"seq_ctc_{layer_idx}", scale=aux_loss_scales[i], use_normalized_loss=True)
        else:
            raise ValueError(f"invalid use_normalized_loss {use_normalized_loss!r}")

    batch_dims = data.remaining_dims(data_spatial_dim)
    input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
    )
    targets_w_eos, _ = rf.pad(
        targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx, out_dims=[targets_w_eos_spatial_dim]
    )
    if text_augment:
        input_labels, targets_w_eos, targets_w_eos_spatial_dim = rf.cond(
            rf.get_run_ctx().train_flag,
            lambda: text_augment(
                input_labels=input_labels,
                targets_w_eos=targets_w_eos,
                spatial_dim=targets_w_eos_spatial_dim,
                exclude_labels={model.bos_idx, model.eos_idx},
            ),
            lambda: (input_labels, targets_w_eos, targets_w_eos_spatial_dim),
        )

    collected_outputs = CollectOutputsDict(
        allowed_key_patterns=[str(layer_idx - 1) for layer_idx in dec_aux_loss_layers]
    )
    logits, _ = model.decoder(
        input_labels,
        spatial_dim=targets_w_eos_spatial_dim,
        encoder=enc,
        state=model.decoder.default_initial_state(batch_dims=batch_dims),
        collected_outputs=collected_outputs,
    )
    dec_aux_logits = {}
    for layer_idx in dec_aux_loss_layers:
        norm = getattr(model, f"dec_aux_final_layer_norm_{layer_idx}")
        linear = getattr(model, f"dec_aux_logits_{layer_idx}")
        out = collected_outputs[str(layer_idx - 1)]
        dec_aux_logits[layer_idx] = linear(norm(out))

    targets_packed, pack_dim = rf.pack_padded(
        targets_w_eos, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False
    )
    for postfix, scale, logits_ in [("", aed_loss_scale, logits)] + [
        (f"_{k}", dec_aux_loss_scales[i], dec_aux_logits[k]) for i, k in enumerate(dec_aux_loss_layers)
    ]:
        logits_packed, _ = rf.pack_padded(
            logits_, dims=batch_dims + [targets_w_eos_spatial_dim], enforce_sorted=False, out_dim=pack_dim
        )

        if not model.out_eos_separated:  # joint distrib, std case
            log_prob = rf.log_softmax(logits_packed, axis=model.target_dim)
        else:  # eos separated
            log_prob = log_probs_with_eos_separated(logits_packed, target_dim=model.target_dim, eos_idx=model.eos_idx)
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, label_smoothing, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_packed, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        if use_normalized_loss in ("none", "frames"):
            loss.mark_as_loss(
                f"ce{postfix}",
                scale=scale,
                use_normalized_loss={"none": False, "frames": True}[use_normalized_loss],
            )
        elif use_normalized_loss == "seqs":
            loss.mark_as_loss(f"ce{postfix}", scale=0)  # don't use this for training directly, just for reporting
            loss_ = rf.pad_packed(loss, dims=batch_dims + [targets_w_eos_spatial_dim], in_dim=pack_dim)
            seq_loss = rf.reduce_sum(loss_, axis=targets_w_eos_spatial_dim)
            seq_loss.mark_as_loss(f"seq_ce{postfix}", scale=scale, use_normalized_loss=True)
        else:
            raise ValueError(f"invalid use_normalized_loss {use_normalized_loss!r}")

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_packed
        frame_error.mark_as_loss(name=f"fer{postfix}", as_error=True)


train_def: TrainDef[Model]
train_def.learning_rate_control_error_measure = "ce"
