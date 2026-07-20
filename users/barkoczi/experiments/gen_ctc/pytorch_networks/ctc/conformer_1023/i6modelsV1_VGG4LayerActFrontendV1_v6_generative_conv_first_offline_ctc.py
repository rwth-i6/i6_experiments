import torch

from i6_experiments.users.barkoczi.experiments.gen_ctc.loss.generative_loss import generative_nce

from .i6modelsV1_VGG4LayerActFrontendV1_v6_generative_conv_first_v2 import (
    Model,
    prior_finish_hook,
    prior_init_hook,
    prior_step,
)


def train_step(*, model: Model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    frozen_targets = data["ctc_soft_targets"]
    frozen_target_len = data["ctc_soft_targets:size1"].to(torch.long)

    log_probs, output_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)
    if frozen_targets.shape[-1] != log_probs.shape[-1]:
        raise ValueError(
            f"Frozen CTC target dimension {frozen_targets.shape[-1]} does not match "
            f"model output dimension {log_probs.shape[-1]}"
        )

    shared_time_dim = min(log_probs.shape[1], frozen_targets.shape[1])
    if shared_time_dim == 0:
        return
    valid_len = torch.minimum(output_len, frozen_target_len.to(output_len.device))
    valid_len = torch.clamp(valid_len, max=shared_time_dim)
    frozen_targets = frozen_targets[:, :shared_time_dim].to(
        device=log_probs.device,
        dtype=log_probs.dtype,
    )

    loss = generative_nce(
        log_probs[:, :shared_time_dim],
        frozen_targets,
        sampling_type=model.cfg.sampling_type,
        seq_len=valid_len,
        sampling_ratio=model.cfg.sampling_ratio,
        share_samples=model.cfg.share_samples,
        ratio_corrector=model.cfg.ratio_corrector,
    )
    inv_norm_factor = torch.clamp(valid_len.sum().to(dtype=torch.float32), min=1.0)
    run_ctx.mark_as_loss(
        name="frozen_ctc_generative_nce",
        loss=loss.sum(),
        inv_norm_factor=inv_norm_factor,
    )
