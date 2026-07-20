from dataclasses import dataclass

import torch

from i6_experiments.users.barkoczi.experiments.gen_ctc.loss.fixed_ctc_loss import torch_ctc_fixed_grad


@dataclass
class ForwardConfig:
    output_filename: str = "ctc_soft_targets.hdf"
    num_classes: int = 80
    storage_dtype: str = "float16"


def forward_init_hook(run_ctx, **kwargs):
    from returnn.datasets.hdf import SimpleHDFWriter

    run_ctx.config = ForwardConfig(**kwargs.get("config", {}))
    if run_ctx.config.storage_dtype not in {"float16", "float32"}:
        raise ValueError(f"Unsupported storage dtype {run_ctx.config.storage_dtype!r}")
    run_ctx.hdf_writer = SimpleHDFWriter(
        run_ctx.config.output_filename,
        dim=run_ctx.config.num_classes,
        ndim=2,
    )


def forward_finish_hook(run_ctx, **kwargs):
    run_ctx.hdf_writer.close()


def _normalize_seq_tags(seq_tags):
    return [tag.decode("utf8") if isinstance(tag, (bytes, bytearray)) else str(tag) for tag in seq_tags]


def forward_step(*, model, data, run_ctx, **kwargs):
    raw_audio = data["raw_audio"]
    raw_audio_len = data["raw_audio:size1"].to("cpu")
    labels = data["labels"]
    labels_len = data["labels:size1"].to("cpu")

    model.eval()
    with torch.no_grad():
        log_probs, output_len = model(raw_audio=raw_audio, raw_audio_len=raw_audio_len)

    # The fixed-gradient CTC loss has gradient -p(path label at t | x, transcript).
    # Detaching here avoids retaining the encoder graph while preserving that exact posterior.
    ctc_scores = log_probs.detach().requires_grad_(True)
    with torch.enable_grad():
        ctc_loss = torch_ctc_fixed_grad(
            ctc_scores.permute(1, 0, 2),
            labels,
            input_lengths=output_len.cpu(),
            target_lengths=labels_len,
            blank=model.cfg.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )
        soft_targets = -torch.autograd.grad(ctc_loss, ctc_scores)[0]

    if soft_targets.shape[-1] != run_ctx.config.num_classes:
        raise ValueError(
            f"Expected {run_ctx.config.num_classes} CTC classes, got {soft_targets.shape[-1]}"
        )
    run_ctx.hdf_writer.insert_batch(
        inputs=soft_targets.detach().cpu().numpy().astype(run_ctx.config.storage_dtype, copy=False),
        seq_len=output_len.detach().cpu().to(torch.long).tolist(),
        seq_tag=_normalize_seq_tags(data["seq_tag"]),
    )
