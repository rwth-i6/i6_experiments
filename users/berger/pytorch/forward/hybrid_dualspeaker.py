import torch
from returnn.tensor.tensor_dict import TensorDict


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, **kwargs):
    audio_features_prim = extern_data["features_primary"].raw_tensor
    audio_features_prim_len = extern_data["features_primary"].dims[1].dyn_size_ext.raw_tensor
    assert audio_features_prim is not None
    assert audio_features_prim_len is not None

    audio_features = audio_features_prim

    try:
        audio_features_sec = extern_data["features_secondary"].raw_tensor
    except KeyError:
        audio_features_sec = None

    try:
        audio_features_mix = extern_data["features_mix"].raw_tensor
    except KeyError:
        audio_features_mix = None

    if audio_features_sec is not None:
        audio_features = torch.concat([audio_features, audio_features_sec], axis=2)
    if audio_features_mix is not None:
        audio_features = torch.concat([audio_features, audio_features_mix], axis=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_probs, out_seq_len = model(
        primary_audio_features=audio_features.to(device),
        audio_features_len=audio_features_prim_len.to(device),
        secondary_audio_features=None,
        mix_audio_features=None,
    )

    import returnn.frontend as rf

    run_ctx = rf.get_run_ctx()
    if run_ctx.expected_outputs is not None:
        run_ctx.expected_outputs["log_probs"].dims[1].dyn_size_ext.raw_tensor = out_seq_len
    run_ctx.mark_as_output(log_probs, name="log_probs")
