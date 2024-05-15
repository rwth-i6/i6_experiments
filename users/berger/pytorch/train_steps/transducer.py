import torch
from returnn.tensor.tensor_dict import TensorDict
from i6_experiments.users.berger.pytorch.helper_functions import map_tensor_to_minus1_plus1_interval


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, blank_idx: int = 0, **kwargs):
    from returnn.extern_private.BergerMonotonicRNNT.monotonic_rnnt.pytorch_binding import monotonic_rnnt_loss

    audio_features = extern_data["data"].raw_tensor
    assert audio_features is not None
    audio_features = audio_features.float()
    # audio_features = map_tensor_to_minus1_plus1_interval(audio_features)

    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_feature_lengths = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_feature_lengths is not None
    audio_feature_lengths = audio_feature_lengths.to(device="cuda")

    assert extern_data["classes"].raw_tensor is not None
    targets = extern_data["classes"].raw_tensor.to(dtype=torch.int32)

    assert extern_data["classes"].dims[1].dyn_size_ext is not None
    target_lengths = extern_data["classes"].dims[1].dyn_size_ext.raw_tensor
    assert target_lengths is not None
    target_lengths = target_lengths.to(device="cuda")

    targets_padded = torch.nn.functional.pad(targets, (1, 0), mode="constant", value=blank_idx)
    targets_padded_lengths = target_lengths + 1

    model_logits, input_lengths, _, _ = model(
        sources=audio_features,
        source_lengths=audio_feature_lengths,
        targets=targets_padded,
        target_lengths=targets_padded_lengths,
    )

    loss = monotonic_rnnt_loss(
        acts=model_logits.to(dtype=torch.float32),
        labels=targets,
        input_lengths=input_lengths,
        label_lengths=target_lengths,
        blank_label=blank_idx,
    )

    import returnn.frontend as rf

    rf.get_run_ctx().mark_as_loss(name="MonoRNNT", loss=loss)
