import torch
from returnn.torch.context import RunCtx


def train_step(*, model: torch.nn.Module, data: dict, run_ctx: RunCtx, **kwargs):
    audio_features = data["data"].float()  # [B, T', F]
    audio_features_len = data["data:size1"]  # [B]

    targets = data["classes"]  # [B, S]
    targets_len = data["classes:size1"]  # [B]

    log_probs, sequence_lengths = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, V]

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=sequence_lengths,
        target_lengths=targets_len,
        blank=0,
        reduction="sum",
        zero_infinity=True,
    )

    num_targets = torch.sum(targets_len)
    run_ctx.mark_as_loss(name="ctc", loss=loss, inv_norm_factor=num_targets)
