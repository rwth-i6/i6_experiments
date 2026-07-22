"""
Consistency-Regularized CTC (arxiv 2410.05101) train step, locally corrected.

Copy of ``cr_ctc_training`` from :mod:`exp2024_10_16_consistency_reg_ctc`, with the one
fix it is missing: the input is expanded to two augmented views (``branch_dim``), so
``targets`` must be expanded to the same ``branch_dim`` -- otherwise the AED aux decoder's
cross-attention query lacks the branch dim the encoder keys carry (returnn#1636).

Kept separate from the main recipe so the existing n12 CR checkpoints are not re-hashed.
"""

from __future__ import annotations

import returnn.frontend as rf
from returnn.tensor import Tensor, Dim
from returnn.frontend.decoder.transformer import TransformerDecoder

from .exp2024_10_16_consistency_reg_ctc import _cr_loss

__setup_base_name__ = "exp2026_05_25_align_behavior_crctc"


def cr_ctc_training_fixed(*, model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """CR-CTC train step with the targets branch-expansion fix. Run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    cr_loss_scale = config.float("cr_loss_scale", 0.2)
    cr_loss_on_aux_probs = config.bool("cr_loss_on_aux_probs", False)
    use_fixed_ctc_grad = config.typed_value("use_fixed_ctc_grad", False)

    ctc_loss = rf.ctc_loss
    if use_fixed_ctc_grad:
        from i6_experiments.users.zeyer.nn_rf.torch_ctc_fixed_grad import ctc_loss_fixed_grad

        assert use_fixed_ctc_grad == "v2"  # v2 has the fix for scaled/normalized CTC loss
        ctc_loss = ctc_loss_fixed_grad

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    branch_dim = Dim(2, name="branch")
    data = rf.expand_dim(data, dim=branch_dim)
    # Fix (returnn#1636): targets must carry the same branch dim as the encoder,
    # so the AED decoder cross-attention query and keys share it.
    targets = rf.expand_dim(targets, dim=branch_dim)

    collected_outputs = {} if aux_loss_layers else None
    logits, enc, enc_spatial_dim = model(data, in_spatial_dim=data_spatial_dim, collected_outputs=collected_outputs)
    for dim in enc_spatial_dim.dyn_size_ext.dims:
        if dim not in data_spatial_dim.dyn_size_ext.dims:
            enc_spatial_dim.dyn_size_ext = rf.gather(enc_spatial_dim.dyn_size_ext, axis=dim, indices=0)
    if aux_loss_layers:
        for i, layer_idx in enumerate(aux_loss_layers):
            if layer_idx > len(model.encoder.layers):
                continue
            linear = getattr(model, f"enc_aux_logits_{layer_idx}")
            aux_logits = linear(collected_outputs[str(layer_idx - 1)])
            aux_log_probs = model.log_probs_wb_from_logits(aux_logits)
            aux_loss = ctc_loss(
                logits=aux_log_probs,
                logits_normalized=True,
                targets=targets,
                input_spatial_dim=enc_spatial_dim,
                targets_spatial_dim=targets_spatial_dim,
                blank_index=model.blank_idx,
            )
            aux_loss.mark_as_loss(
                f"ctc_{layer_idx}",
                scale=aux_loss_scales[i] * 0.5,
                custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
                use_normalized_loss=use_normalized_loss,
            )
            if cr_loss_on_aux_probs:
                _cr_loss(
                    f"consistency_{layer_idx}",
                    aux_log_probs,
                    branch_dim=branch_dim,
                    wb_target_dim=model.wb_target_dim,
                    scale=cr_loss_scale * aux_loss_scales[i],
                    use_normalized_loss=use_normalized_loss,
                )

    log_probs = model.log_probs_wb_from_logits(logits)
    loss = ctc_loss(
        logits=log_probs,
        logits_normalized=True,
        targets=targets,
        input_spatial_dim=enc_spatial_dim,
        targets_spatial_dim=targets_spatial_dim,
        blank_index=model.blank_idx,
    )
    loss.mark_as_loss(
        "ctc",
        scale=0.5,
        custom_inv_norm_factor=targets_spatial_dim.get_size_tensor(),
        use_normalized_loss=use_normalized_loss,
    )

    _cr_loss(
        "consistency",
        log_probs,
        branch_dim=branch_dim,
        wb_target_dim=model.wb_target_dim,
        scale=cr_loss_scale,
        use_normalized_loss=use_normalized_loss,
    )

    if model.decoder:
        # noinspection PyTypeChecker
        decoder: TransformerDecoder = model.decoder

        input_labels, (targets_w_eos_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(1, 0)], value=model.bos_idx
        )
        targets_w_eos, _ = rf.pad(
            targets,
            axes=[targets_spatial_dim],
            padding=[(0, 1)],
            value=model.eos_idx,
            out_dims=[targets_w_eos_spatial_dim],
        )

        batch_dims = data.remaining_dims(data_spatial_dim)
        logits, _ = model.decoder(
            input_labels,
            spatial_dim=targets_w_eos_spatial_dim,
            encoder=decoder.transform_encoder(enc, axis=enc_spatial_dim),
            state=model.decoder.default_initial_state(batch_dims=batch_dims),
        )

        log_prob = rf.log_softmax(logits, axis=model.target_dim)
        log_prob = rf.label_smoothed_log_prob_gradient(log_prob, 0.1, axis=model.target_dim)
        loss = rf.cross_entropy(
            target=targets_w_eos, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
        )
        loss.verify_out_shape(set(batch_dims) | {branch_dim, targets_w_eos_spatial_dim})
        loss.mark_as_loss("aed_ce", scale=aed_loss_scale * 0.5, use_normalized_loss=use_normalized_loss)

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_w_eos
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


cr_ctc_training_fixed.learning_rate_control_error_measure = "ctc"
