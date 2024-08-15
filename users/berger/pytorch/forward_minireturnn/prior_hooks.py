import torch
from returnn.torch.context import RunCtx


def prior_init_hook(run_ctx: RunCtx, **kwargs):
    run_ctx.sum_probs = None
    run_ctx.sum_frames = 0


def prior_finish_hook(run_ctx: RunCtx, **kwargs):
    all_probs = run_ctx.sum_probs.detach().cpu().numpy()
    all_frames = run_ctx.sum_frames.detach().cpu().numpy()
    average_probs = all_probs / all_frames
    log_average_probs = np.log(average_probs)
    print("Prior sum in std-space (should be close to 1.0):", np.sum(average_probs))
    with open("prior.txt", "w") as f:
        np.savetxt(f, log_average_probs, delimiter=" ")
    print("Saved prior in prior.txt in +log space.")


def prior_step(*, model: torch.nn.Module, data: dict, run_ctx: RunCtx, **kwargs):
    audio_features = data["data"].float()  # [B, T', F]
    audio_features_len = data["data:size1"]  # [B]

    log_probs, sequence_lengths = model(
        audio_features=audio_features,
        audio_features_len=audio_features_len.to("cuda"),
    )

    probs = torch.exp(log_probs)

    run_ctx.sum_frames = run_ctx.sum_frames + torch.sum(sequence_lengths)
    if run_ctx.sum_probs is None:
        run_ctx.sum_probs = torch.sum(probs, dim=(0, 1))
    else:
        run_ctx.sum_probs += torch.sum(probs, dim=(0, 1))
