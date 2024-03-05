import torch
from returnn.tensor.tensor_dict import TensorDict


def train_step(*, model: torch.nn.Module, extern_data: TensorDict, simple_loss_scale: float = 0.5, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    assert extern_data["data"].dims[1].dyn_size_ext is not None
    audio_features_len_rf = extern_data["data"].dims[1].dyn_size_ext
    assert audio_features_len_rf is not None
    audio_features_len = audio_features_len_rf.raw_tensor
    assert audio_features_len is not None

    assert extern_data["targets"].raw_tensor is not None
    targets = extern_data["targets"].raw_tensor.long()
    assert extern_data["targets"].dims[1].dyn_size_ext is not None
    targets_len = extern_data["targets"].dims[1].dyn_size_ext.raw_tensor
    assert targets_len is not None

    model.train()

    simple_loss, pruned_loss = model.forward_with_pruned_loss(
        features=audio_features.to(device="cuda"),
        features_len=audio_features_len.to(device="cuda"),
        targets=targets.to(device="cuda"),
        targets_len=targets_len.to(device="cuda"),
    )

    from returnn.tensor import batch_dim
    import returnn.frontend as rf

    num_frames = rf.reduce_sum(audio_features_len_rf, axis=batch_dim)

    rf.get_run_ctx().mark_as_loss(
        name="RNNT simple", custom_inv_norm_factor=num_frames, loss=simple_loss, scale=simple_loss_scale
    )
    rf.get_run_ctx().mark_as_loss(name="RNNT pruned", custom_inv_norm_factor=num_frames, loss=pruned_loss)
