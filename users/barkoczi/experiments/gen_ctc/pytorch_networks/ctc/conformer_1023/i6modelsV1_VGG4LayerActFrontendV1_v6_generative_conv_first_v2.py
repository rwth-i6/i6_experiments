import torch

from .i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first import (
    Model,
    mask_tensor,
    prior_finish_hook,
    prior_init_hook,
    train_step,
)


def prior_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"]
    log_probs, audio_features_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    probs = torch.exp(log_probs)
    valid_mask = mask_tensor(probs, audio_features_len).unsqueeze(-1)
    batch_sum_probs = torch.sum(probs * valid_mask, dim=(0, 1))
    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(audio_features_len)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = batch_sum_probs
    else:
        run_ctx.sum_probs += batch_sum_probs
