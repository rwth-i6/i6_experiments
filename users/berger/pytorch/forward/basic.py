import torch
from returnn.tensor.tensor_dict import TensorDict
from ..helper_functions import map_tensor_to_minus1_plus1_interval


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features = extern_data["data"].raw_tensor
    assert audio_features is not None
    audio_features = map_tensor_to_minus1_plus1_interval(audio_features)

    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_len is not None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_probs, out_seq_len = model(
        audio_features=audio_features.to(device),
        audio_features_len=audio_features_len.to(device),
    )  # [B, T, F]

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext.raw_tensor = out_seq_len
    run_ctx.mark_as_output(log_probs, name="log_probs")
