import torch

from returnn_mini.torch.context import RunCtx


def train_step(*, model: torch.nn.Module, data: dict, run_ctx: RunCtx, **kwargs):
    assert "data" in data
    assert "data:size1" in data
    assert "classes" in data
    assert "classes:size1" in data

    audio_features = data["data"]
    audio_features_len = data["data:size1"]

    targets = data["classes"]
    targets_len = data["classes:size1"]

    log_probs = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    downsample_factor = round(audio_features.shape[1] / log_probs.shape[1])
    sequence_lengths = torch.ceil(audio_features_len / downsample_factor)

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )

    run_ctx.mark_as_loss(name="CTC", loss=loss)
