import torch
from returnn.tensor.tensor_dict import TensorDict


def map_tensor_to_minus1_plus1_interval(tensor: torch.Tensor) -> torch.Tensor:
    if torch.is_floating_point(tensor):
        return tensor

    dtype = tensor.dtype
    info = torch.iinfo(dtype)
    min_val = info.min
    max_val = info.max

    return 2.0 * (tensor.float() - min_val) / (max_val - min_val) - 1.0

def train_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features = audio_features.squeeze(-1)
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)
    assert extern_data["data"].dims[1].dyn_size_ext is not None

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()

    targets_len_rf = extern_data["targets"].dims[1].dyn_size_ext
    assert targets_len_rf is not None
    targets_len = targets_len_rf.raw_tensor
    assert targets_len is not None

    log_probs, sequence_lengths = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )

    from returnn.tensor import batch_dim
    import returnn.frontend as rf

    rf.get_run_ctx().mark_as_loss(
        name="CTC", loss=loss, custom_inv_norm_factor=rf.reduce_sum(targets_len_rf, axis=batch_dim)
    )