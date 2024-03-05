from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    log_probs, audio_features_len = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )  # [B, T, F]

    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")
