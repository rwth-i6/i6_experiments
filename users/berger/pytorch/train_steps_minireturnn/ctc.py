import torch
from returnn.torch.context import RunCtx


def train_step(*, model: torch.nn.Module, data: dict, run_ctx: RunCtx, blank_idx: int = 0, **kwargs):
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
        blank=blank_idx,
        reduction="sum",
        zero_infinity=True,
    )

    num_targets = torch.sum(targets_len)
    run_ctx.mark_as_loss(name="ctc", loss=loss, inv_norm_factor=num_targets)


def train_step_nick(*, model, data, run_ctx, **kwargs):
    raw_audio = data["data"]  # [B, T', F]
    raw_audio_len = data["data:size1"].to("cpu")  # [B]

    labels = data["classes"]  # [B, N] (sparse)
    labels_len = data["classes:size1"]  # [B, N]

    logprobs, audio_features_len = model(
        raw_audio=raw_audio,
        raw_audio_len=raw_audio_len,
    )
    transposed_logprobs = torch.permute(logprobs, (1, 0, 2))  # CTC needs [T, B, F]
    ctc_loss = torch.nn.functional.ctc_loss(
        transposed_logprobs,
        labels,
        input_lengths=audio_features_len,
        target_lengths=labels_len,
        blank=model.cfg.label_target_size,
        reduction="sum",
        zero_infinity=True,
    )
    num_phonemes = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="ctc", loss=ctc_loss, inv_norm_factor=num_phonemes)
