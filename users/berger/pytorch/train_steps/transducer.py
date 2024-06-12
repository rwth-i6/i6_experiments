from typing import Dict, Optional

import torch
from returnn.tensor.tensor_dict import TensorDict

from i6_experiments.users.berger.pytorch.models.conformer_transducer_v2 import FFNNTransducer, FFNNTransducerViterbi


def train_step(
    *,
    model: FFNNTransducer,
    extern_data: TensorDict,
    blank_idx: int = 0,
    enc_loss_scales: Optional[Dict[int, float]] = None,
    **_,
):
    import returnn.frontend as rf
    from pytorch_binding.monotonic_rnnt_op import monotonic_rnnt_loss
    from returnn.tensor import batch_dim

    if enc_loss_scales is None:
        enc_loss_scales = {}

    sources = extern_data["data"].raw_tensor
    assert sources is not None
    sources = sources.float()

    assert extern_data["data"].dims[1].dyn_size_ext is not None
    source_lengths = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert source_lengths is not None
    source_lengths = source_lengths.to(device="cuda")

    assert extern_data["classes"].raw_tensor is not None
    targets = extern_data["classes"].raw_tensor.to(dtype=torch.int32)

    target_lengths_rf = extern_data["classes"].dims[1].dyn_size_ext
    assert target_lengths_rf is not None

    target_lengths = target_lengths_rf.raw_tensor
    assert target_lengths is not None
    target_lengths = target_lengths.to(device="cuda")

    loss_norm_factor = rf.reduce_sum(target_lengths_rf, axis=batch_dim)

    model_logits, intermediate_logits, source_lengths, _, _ = model.forward(
        sources=sources,
        source_lengths=source_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )

    loss = monotonic_rnnt_loss(
        acts=model_logits.to(dtype=torch.float32),
        labels=targets,
        input_lengths=source_lengths,
        label_lengths=target_lengths,
        blank_label=blank_idx,
    )
    rf.get_run_ctx().mark_as_loss(name="monotonic_rnnt", loss=loss, custom_inv_norm_factor=loss_norm_factor)

    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets, target_lengths.cpu(), batch_first=True, enforce_sorted=False
    )

    for layer_idx, scale in enc_loss_scales.items():
        logits = intermediate_logits[layer_idx]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # [B, T, C]
        log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, C]

        loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=source_lengths,
            target_lengths=target_lengths,
            blank=blank_idx,
            reduction="sum",
            zero_infinity=True,
        )

        rf.get_run_ctx().mark_as_loss(
            name=f"CTC_enc-{layer_idx}",
            loss=loss,
            scale=scale,
            custom_inv_norm_factor=loss_norm_factor,
        )

        predictions = torch.argmax(log_probs, dim=-1)
        predictions_packed = torch.nn.utils.rnn.pack_padded_sequence(
            predictions, target_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        num_incorrect_frames = torch.sum((targets_packed.data != predictions_packed.data))

        rf.get_run_ctx().mark_as_loss(
            name=f"FER_enc-{layer_idx}",
            loss=num_incorrect_frames,
            custom_inv_norm_factor=loss_norm_factor,
            as_error=True,
        )


def train_step_viterbi(
    *,
    model: FFNNTransducerViterbi,
    extern_data: TensorDict,
    enc_loss_scales: Optional[Dict[int, float]] = None,
    **_,
):
    import returnn.frontend as rf
    from returnn.tensor import batch_dim

    if enc_loss_scales is None:
        enc_loss_scales = {}

    sources = extern_data["data"].raw_tensor
    assert sources is not None
    sources = sources.float()

    assert extern_data["classes"].raw_tensor is not None
    targets = extern_data["classes"].raw_tensor.to(dtype=torch.int32)

    assert extern_data["data"].dims[1].dyn_size_ext is not None
    source_lengths = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert source_lengths is not None
    source_lengths = source_lengths.to(device="cuda")

    target_lengths_rf = extern_data["classes"].dims[1].dyn_size_ext
    assert target_lengths_rf is not None

    target_lengths = target_lengths_rf.raw_tensor
    assert target_lengths is not None
    target_lengths = target_lengths.to(device="cuda")

    loss_norm_factor = rf.reduce_sum(target_lengths_rf, axis=batch_dim)

    model_logits, intermediate_logits, _, _ = model.forward(
        sources=sources,
        source_lengths=source_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )

    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets.long(), target_lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    for logits, scale, suffix in [(model_logits, 1.0, "final")] + [
        (intermediate_logits[layer_idx], scale, f"enc-{layer_idx}") for layer_idx, scale in enc_loss_scales.items()
    ]:
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, F]
        log_probs = torch.transpose(log_probs, 1, 2)  # [B, F, T]

        loss = torch.nn.functional.cross_entropy(
            input=log_probs, target=targets_masked, ignore_index=-100, reduction="sum"
        )

        predictions = torch.argmax(logits, dim=-1)
        predictions_packed = torch.nn.utils.rnn.pack_padded_sequence(
            predictions.long(), target_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        num_incorrect_frames = torch.sum((targets_packed.data != predictions_packed.data))

        rf.get_run_ctx().mark_as_loss(
            name=f"CE_{suffix}", loss=loss, scale=scale, custom_inv_norm_factor=loss_norm_factor
        )
        rf.get_run_ctx().mark_as_loss(
            name=f"FER_{suffix}", loss=num_incorrect_frames, custom_inv_norm_factor=loss_norm_factor, as_error=True
        )


def train_step_align_restrict(
    *,
    model: FFNNTransducer,
    extern_data: TensorDict,
    max_distance_from_alignment: int,
    blank_idx: int = 0,
    enc_loss_scales: Optional[Dict[int, float]] = None,
    **_,
):
    import returnn.frontend as rf
    from pytorch_binding.monotonic_rnnt_op import monotonic_rnnt_loss
    from returnn.tensor import batch_dim

    if enc_loss_scales is None:
        enc_loss_scales = {}

    sources = extern_data["data"].raw_tensor
    assert sources is not None
    sources = sources.float()

    assert extern_data["data"].dims[1].dyn_size_ext is not None
    source_lengths = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert source_lengths is not None
    source_lengths = source_lengths.to(device="cuda")

    assert extern_data["classes"].raw_tensor is not None
    alignments = extern_data["classes"].raw_tensor.to(torch.int32)  # [B, T]
    alignment_lengths_rf = extern_data["classes"].dims[1].dyn_size_ext
    assert alignment_lengths_rf is not None

    target_lengths = (alignments != blank_idx).sum(dim=1)  # [B]
    target_lengths = target_lengths.to(device="cuda", dtype=torch.int32)
    target_lengths_rf = rf.Tensor("target_lengths", dims=[batch_dim], dtype="int32", raw_tensor=target_lengths)

    max_target_length = target_lengths.max().item()
    targets = torch.zeros((target_lengths.size(0), max_target_length), dtype=alignments.dtype, device=alignments.device)

    for i in range(targets.size(0)):
        non_blanks = alignments[i][alignments[i] != blank_idx]
        targets[i, : non_blanks.size(0)] = non_blanks

    target_loss_norm_factor = rf.reduce_sum(target_lengths_rf, axis=batch_dim)
    alignment_loss_norm_factor = rf.reduce_sum(alignment_lengths_rf, axis=batch_dim)

    model_logits, intermediate_logits, source_lengths, _, _ = model.forward(
        sources=sources,
        source_lengths=source_lengths,
        targets=targets,
        target_lengths=target_lengths,
    )

    loss = monotonic_rnnt_loss(
        acts=model_logits.to(dtype=torch.float32),
        labels=targets,
        input_lengths=source_lengths,
        label_lengths=target_lengths,
        alignment=alignments,
        max_distance_from_alignment=max_distance_from_alignment,
        blank_label=blank_idx,
    )
    rf.get_run_ctx().mark_as_loss(
        name=f"monotonic_rnnt_restrict-{max_distance_from_alignment}",
        loss=loss,
        custom_inv_norm_factor=target_loss_norm_factor,
    )

    alignments_packed = torch.nn.utils.rnn.pack_padded_sequence(
        alignments, source_lengths.cpu(), batch_first=True, enforce_sorted=False
    )
    alignments_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(
        alignments_packed, batch_first=True, padding_value=-100
    )

    for logits, scale, suffix in [
        (intermediate_logits[layer_idx], scale, f"enc-{layer_idx}") for layer_idx, scale in enc_loss_scales.items()
    ]:
        log_probs = torch.log_softmax(logits, dim=-1)  # [B, T, F]
        log_probs = torch.transpose(log_probs, 1, 2)  # [B, F, T]

        loss = torch.nn.functional.cross_entropy(
            input=log_probs, target=alignments_masked.long(), ignore_index=-100, reduction="sum"
        )

        predictions = torch.argmax(logits, dim=-1)
        predictions_packed = torch.nn.utils.rnn.pack_padded_sequence(
            predictions.long(), source_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        num_incorrect_frames = torch.sum((alignments_packed.data != predictions_packed.data))

        rf.get_run_ctx().mark_as_loss(
            name=f"CE_{suffix}", loss=loss, scale=scale, custom_inv_norm_factor=alignment_loss_norm_factor
        )
        rf.get_run_ctx().mark_as_loss(
            name=f"FER_{suffix}",
            loss=num_incorrect_frames,
            custom_inv_norm_factor=alignment_loss_norm_factor,
            as_error=True,
        )
