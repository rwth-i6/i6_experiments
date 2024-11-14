"""
Consistency-Regularized CTC <https://arxiv.org/pdf/2410.05101>
"""

from __future__ import annotations
from i6_experiments.users.zeyer.model_interfaces import ModelDef, ModelDefWithCfg, RecogDef, TrainDef
from returnn.tensor import Tensor, Dim
import returnn.frontend as rf
from returnn.frontend.decoder.transformer import TransformerDecoder
from returnn.frontend.encoder.conformer import ConformerEncoderLayer, ConformerPositionwiseFeedForward

# noinspection PyProtectedMember
from .exp2024_04_23_baselines.ctc import (
    Model,
    train_exp,
    config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
    _get_cfg_lrlin_oclr_by_bs_nep,
    speed_pert_librosa_config,
)

# noinspection PyProtectedMember
from .exp2024_04_23_baselines.configs import _get_cfg_lrlin_oclr_by_bs_nep_v2


__setup_base_name__ = "exp2024_10_16_consistency_reg_ctc"
# __setup_root_prefix__ = __setup_base_name__  # ...


def py():
    for opts in [
        # Baseline (n12) has {"dev-clean": 2.35, "dev-other": 5.65, "test-clean": 2.66, "test-other": 5.94}.
        # v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        {
            "name": "v6-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs15k-accgrad1-mgpu4-pavg100-wd1e_2"
            "-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001",
            "num_enc_layers": 12,
            "batch_size": 15_000,
        },
        # Baseline (n16) has {"dev-clean": 2.26, "dev-other": 5.44, "test-clean": 2.5, "test-other": 5.62}.
        # v6-n16-relPosAttDef-noBias-aedLoss-bhv20-11gb-f32-bs10k-accgrad1-mgpu4-pavg100-wd1e_2-lrlin1e_5_295k-featBN-speedpertV2-spm10k-bpeSample001
        # {"num_enc_layers": 16, "batch_size": 10_000},
    ]:
        for cr_ctc in [None, {"cr_loss_scale": 0.2}, {"cr_loss_scale": 0.2, "aed_loss_bug_fix": True}]:
            # TODO also adapt specaug for CR...
            use_cr_ctc = cr_ctc is not None
            if use_cr_ctc:
                name = f"n{opts['num_enc_layers']}"
                name += f"-crLoss{cr_ctc['cr_loss_scale']}"
                name += "-aedLossBug" if not cr_ctc.get("aed_loss_bug_fix") else ""
            else:
                name = opts["name"]
            train_exp(
                name,
                config_11gb_v6_f32_accgrad1_mgpu4_pavg100_wd1e_4,
                train_def=cr_ctc_training if use_cr_ctc else None,
                model_config={
                    "enc_conformer_layer": rf.build_dict(
                        ConformerEncoderLayer,
                        ff=rf.build_dict(
                            ConformerPositionwiseFeedForward, activation=rf.build_dict(rf.relu_square), with_bias=False
                        ),
                        num_heads=8,
                    ),
                    "feature_batch_norm": True,
                    **({"num_enc_layers": opts["num_enc_layers"]} if opts["num_enc_layers"] != 12 else {}),
                },
                config_updates={
                    **(
                        _get_cfg_lrlin_oclr_by_bs_nep_v2((opts["batch_size"] // 2 + 501) // 1000 * 1000, 250)
                        if use_cr_ctc
                        else _get_cfg_lrlin_oclr_by_bs_nep(opts["batch_size"], 500)
                    ),
                    "optimizer.weight_decay": 1e-2,
                    "__train_audio_preprocess": speed_pert_librosa_config,
                    "speed_pert_discrete_values": [0.7, 0.8, 0.9, 1.0, 1.1],
                    "aux_attention_decoder": rf.build_dict(
                        TransformerDecoder, num_layers=6
                    ),  # purely used for training
                    **(cr_ctc if use_cr_ctc else {}),
                },
                vocab="spm10k",
                train_vocab_opts={"other_opts": {"class": "SamplingBytePairEncoding", "breadth_prob": 0.01}},
                # avoid OOM
                env_updates={"PYTORCH_CUDA_ALLOC_CONF": "backend:cudaMallocAsync,expandable_segments:True"},
            )


def cr_ctc_training(*, model: Model, data: Tensor, data_spatial_dim: Dim, targets: Tensor, targets_spatial_dim: Dim):
    """Function is run within RETURNN."""
    from returnn.config import get_global_config

    config = get_global_config()  # noqa
    aux_loss_layers = config.typed_value("aux_loss_layers")
    aux_loss_scales = config.typed_value("aux_loss_scales", ([1.0] * len(aux_loss_layers)) if aux_loss_layers else None)
    aed_loss_scale = config.float("aed_loss_scale", 1.0)
    use_normalized_loss = config.bool("use_normalized_loss", True)
    cr_loss_scale = config.float("cr_loss_scale", 0.2)
    aed_loss_bug_fix = config.bool("aed_loss_bug_fix", False)

    if data.feature_dim and data.feature_dim.dimension == 1:
        data = rf.squeeze(data, axis=data.feature_dim)
    assert not data.feature_dim  # raw audio

    if config.bool("use_eos_postfix", False):
        targets, (targets_spatial_dim,) = rf.pad(
            targets, axes=[targets_spatial_dim], padding=[(0, 1)], value=model.eos_idx
        )

    branch_dim = Dim(2, name="branch")
    data = rf.expand_dim(data, dim=branch_dim)

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
            aux_loss = rf.ctc_loss(
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
            # decoded, decoded_spatial_dim = rf.ctc_greedy_decode(aux_logits, in_spatial_dim=enc_spatial_dim)
            # error = rf.edit_distance(
            #     a=decoded, a_spatial_dim=decoded_spatial_dim, b=targets, b_spatial_dim=targets_spatial_dim
            # )
            # error.mark_as_loss("label", as_error=True, custom_inv_norm_factor=targets_spatial_dim.get_size_tensor())

    log_probs = model.log_probs_wb_from_logits(logits)
    loss = rf.ctc_loss(
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

    assert branch_dim in log_probs.dims
    log_probs_a = rf.gather(log_probs, axis=branch_dim, indices=0)
    log_probs_b = rf.gather(log_probs, axis=branch_dim, indices=1)
    consistency_reg_a = rf.cross_entropy(
        estimated=log_probs_a,
        estimated_type="log-probs",
        target=rf.stop_gradient(rf.exp(log_probs_b)),
        axis=model.wb_target_dim,
    )
    consistency_reg_b = rf.cross_entropy(
        estimated=log_probs_b,
        estimated_type="log-probs",
        target=rf.stop_gradient(rf.exp(log_probs_a)),
        axis=model.wb_target_dim,
    )
    consistency_reg = (consistency_reg_a + consistency_reg_b) * 0.5
    consistency_reg.mark_as_loss("consistency", scale=cr_loss_scale, use_normalized_loss=use_normalized_loss)

    if model.decoder:
        # potentially also other types but just assume
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
        if aed_loss_bug_fix:
            loss = rf.cross_entropy(
                target=targets_w_eos, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
            )
            loss.verify_out_shape(set(batch_dims) | {branch_dim, targets_w_eos_spatial_dim})
        else:
            # Note: this is wrong. targets does not have the correct spatial dim. We get shape [B,branch,S,S+1] here.
            # Also see: https://github.com/rwth-i6/returnn/issues/1636
            loss = rf.cross_entropy(
                target=targets, estimated=log_prob, estimated_type="log-probs", axis=model.target_dim
            )
        loss.mark_as_loss("aed_ce", scale=aed_loss_scale * 0.5, use_normalized_loss=use_normalized_loss)

        best = rf.reduce_argmax(log_prob, axis=model.target_dim)
        frame_error = best != targets_w_eos
        frame_error.mark_as_loss(name="aed_fer", as_error=True)


cr_ctc_training: TrainDef[Model]
cr_ctc_training.learning_rate_control_error_measure = "ctc"
