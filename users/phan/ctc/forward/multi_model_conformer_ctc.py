from returnn.tensor.tensor_dict import TensorDict
import returnn.frontend as rf
import torch


def forward_step(*, model: torch.nn.Module, extern_data: TensorDict, conformer_ctc_name: str, **kwargs):
    """
    Forward the conformer CTC in a MultiModelWrapper
    """
    audio_features = extern_data["data"].raw_tensor
    audio_features_len = extern_data["data"].dims[1].dyn_size_ext.raw_tensor

    log_probs, _, _ = model(
        args = [],
        kwargs = {
            "audio_features": audio_features,
            "audio_features_len": audio_features_len.to("cuda"),
        },
        module=conformer_ctc_name,
        inference=True,
    )  # [B, T, F]

    rf.get_run_ctx().mark_as_output(log_probs, name="log_probs")
