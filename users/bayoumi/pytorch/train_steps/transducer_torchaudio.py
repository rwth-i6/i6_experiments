import torch
from torchaudio.models.rnnt import RNNT
from returnn.tensor.tensor_dict import TensorDict


def train_step(*, model: RNNT, extern_data: TensorDict, blank_id: int, **kwargs):
    from torchaudio.transforms import RNNTLoss

    audio_features = extern_data["data"].raw_tensor  # [B, T, F]
    assert audio_features is not None
    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_features_len_rf = extern_data["data"].dims[1].dyn_size_ext  # [B]
    assert audio_features_len_rf is not None
    audio_features_len = audio_features_len_rf.raw_tensor  # [B]
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.to(torch.int32)  # [B, S]
    assert extern_data["targets"].dims[1].dyn_size_ext is not None
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert targets_len is not None

    device = "cuda"

    targets_prepend = torch.empty((targets.size(0), targets.size(1) + 1), dtype=targets.dtype, device=targets.device)
    targets_prepend[:, 0] = blank_id
    targets_prepend[:, 1:] = targets
    targets_prepend_len = targets_len + 1

    loss_computation = RNNTLoss(blank=blank_id, reduction="sum")

    logits, enc_len, _, _ = model.forward(
        sources=audio_features.to(device=device),
        source_lengths=audio_features_len.to(device=device),
        targets=targets_prepend.to(device=device),
        target_lengths=targets_prepend_len.to(device=device),
    )  # [B, T, S+1, C], [B]

    loss = loss_computation.forward(
        logits=logits,
        targets=targets.to(device=device),
        logit_lengths=enc_len,
        target_lengths=targets_len.to(device=device),
    )

    import returnn.frontend as rf

    B = rf.constant(targets.size(0), dims=[])

    rf.get_run_ctx().mark_as_loss(name="RNNT", custom_inv_norm_factor=B, loss=loss)


def train_step_k2(*, model: RNNT, extern_data: TensorDict, blank_id: int, rnnt_type: str = "regular", **kwargs):
    import fast_rnnt

    audio_features = extern_data["data"].raw_tensor  # [B, T, F]
    assert audio_features is not None
    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_features_len_rf = extern_data["data"].dims[1].dyn_size_ext  # [B]
    assert audio_features_len_rf is not None
    audio_features_len = audio_features_len_rf.raw_tensor  # [B]
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()  # [B, S]
    assert extern_data["targets"].dims[1].dyn_size_ext is not None
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert targets_len is not None

    device = "cuda"

    targets_prepend = torch.empty((targets.size(0), targets.size(1) + 1), dtype=targets.dtype, device=targets.device)
    targets_prepend[:, 0] = blank_id
    targets_prepend[:, 1:] = targets
    targets_prepend_len = targets_len + 1

    logits, enc_len, _, _ = model.forward(
        sources=audio_features.to(device=device),
        source_lengths=audio_features_len.to(device=device),
        targets=targets_prepend.to(device=device),
        target_lengths=targets_prepend_len.to(device=device),
    )  # [B, T, S+1, C], [B]

    boundary = torch.zeros((logits.size(0), 4), dtype=torch.int64, device=device)
    boundary[:, 2] = targets_len.to(device=device)
    boundary[:, 3] = enc_len

    loss = fast_rnnt.rnnt_loss(
        logits=logits,
        symbols=targets.to(device=device),
        boundary=boundary,
        termination_symbol=blank_id,
        rnnt_type=rnnt_type,
        reduction="sum",
    )

    import returnn.frontend as rf

    B = rf.constant(targets.size(0), dims=[])

    rf.get_run_ctx().mark_as_loss(name="RNNT", custom_inv_norm_factor=B, loss=loss)


def train_step_k2_pruned(
    *,
    model: RNNT,
    extern_data: TensorDict,
    blank_id: int,
    lm_only_scale: float = 0.25,
    am_only_scale: float = 0.0,
    prune_range: int = 5,
    simple_loss_scale: float = 0.5,
    rnnt_type: str = "regular",
    **kwargs,
):
    import fast_rnnt

    audio_features = extern_data["data"].raw_tensor  # [B, T, F]
    assert audio_features is not None
    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_features_len_rf = extern_data["data"].dims[1].dyn_size_ext  # [B]
    assert audio_features_len_rf is not None
    audio_features_len = audio_features_len_rf.raw_tensor  # [B]
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()  # [B, S]
    assert extern_data["targets"].dims[1].dyn_size_ext is not None
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor  # [B]
    assert targets_len is not None

    device = "cuda"

    targets_prepend = torch.empty((targets.size(0), targets.size(1) + 1), dtype=targets.dtype, device=targets.device)
    targets_prepend[:, 0] = blank_id
    targets_prepend[:, 1:] = targets
    targets_prepend_len = targets_len + 1

    enc, enc_len = model.transcribe(audio_features.to(device=device), audio_features_len.to(device=device))

    pred, _, _ = model.predict(targets_prepend, targets_prepend_len, state=None)

    boundary = torch.zeros((enc.size(0), 4), dtype=torch.int64, device=device)
    boundary[:, 2] = targets_len.to(device=device)
    boundary[:, 3] = enc_len

    # [B]
    simple_loss, (px_grad, py_grad) = fast_rnnt.rnnt_loss_smoothed(
        lm=pred,
        am=enc,
        symbols=targets.to(device=device),
        termination_symbol=blank_id,
        lm_only_scale=lm_only_scale,
        am_only_scale=am_only_scale,
        boundary=boundary,
        rnnt_type=rnnt_type,
        reduction="sum",
        return_grad=True,
    )

    # [B, T, prune_range]
    ranges = fast_rnnt.get_rnnt_prune_ranges(
        px_grad=px_grad,
        py_grad=py_grad,
        boundary=boundary,
        s_range=prune_range,
    )

    # [B, T, prune_range, C], [B, T, prune_range, C]
    am_pruned, lm_pruned = fast_rnnt.do_rnnt_pruning(am=enc, lm=pred, ranges=ranges)

    # [B, T, prune_range, C]
    logits = model.joiner.forward_samelength(am_pruned, lm_pruned)

    # [B]
    pruned_loss = fast_rnnt.rnnt_loss_pruned(
        logits=logits,
        symbols=targets,
        ranges=ranges,
        termination_symbol=blank_id,
        boundary=boundary,
        rnnt_type=rnnt_type,
        reduction="sum",
    )

    import returnn.frontend as rf

    B = rf.constant(targets.size(0), dims=[])

    rf.get_run_ctx().mark_as_loss(
        name="RNNT simple", custom_inv_norm_factor=B, loss=simple_loss.sum(), scale=simple_loss_scale
    )
    rf.get_run_ctx().mark_as_loss(name="RNNT pruned", custom_inv_norm_factor=B, loss=pruned_loss.sum())
