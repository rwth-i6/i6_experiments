import torch

from returnn_mini.torch.context import RunCtx


def train_step(*, model: torch.nn.Module, run_ctx: RunCtx, data: dict, **kwargs):
    print(data)
    audio_features = data["data"]
    audio_features_len = data["data:size1"]

    targets = data["targets"]
    targets_len = data["targets:size1"]

    log_probs = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, F]

    downsample_factor = round(audio_features.shape[1] / log_probs.shape[0])
    sequence_lengths = torch.ceil(audio_features_len / downsample_factor)
    sequence_lengths = sequence_lengths.type(torch.int32)

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
