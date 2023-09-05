from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    print("Start forward step")
    audio_features_prim = extern_data["features_primary"].raw_tensor
    audio_features_prim_len = extern_data["features_primary"].dims[1].dyn_size_ext.raw_tensor
    audio_features_sec = extern_data["features_secondary"].raw_tensor
    audio_features_mix = extern_data["features_mix"].raw_tensor
    assert audio_features_prim_len is not None

    log_probs = model(
        primary_audio_features=audio_features_prim,
        audio_features_len=audio_features_prim_len.to("cuda"),
        secondary_audio_features=audio_features_sec,
        mix_audio_features=audio_features_mix,
    )

    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")
