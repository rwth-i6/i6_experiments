import torch
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import Import

from .pytorch_modules import MemristorAdapterModel


def get_train_step_import() -> Import:
    return Import(f"{_train_step.__module__}.{_train_step.__name__}", import_as="train_step")


def _train_step(*, model: MemristorAdapterModel, data: dict, run_ctx: RunCtx, **_):
    audio_samples = data["data"]
    audio_samples_size = data["data:size1"].to(device=audio_samples.device)
    targets = data["classes"].to(device=audio_samples.device, dtype=torch.long)
    targets_size = data["classes:size1"].to(device=audio_samples.device)
    loss_norm_factor = targets_size.sum()
    if loss_norm_factor <= 0:
        raise ValueError(f"invalid loss normalization factor {loss_norm_factor} for batch {data.get('seq_tag', None)!r}")

    logits, encoder_states_size = model(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
        targets=targets,
        targets_size=targets_size,
    )

    invalid = (encoder_states_size <= 0) | (targets_size < 0) | (targets_size > encoder_states_size)
    if invalid.any():
        indices = invalid.nonzero(as_tuple=True)[0].tolist()
        seq_tags = [data["seq_tag"][index] if "seq_tag" in data else index for index in indices]
        raise ValueError(
            f"Invalid monotonic RNN-T lengths for {seq_tags!r}: "
            f"encoder={encoder_states_size[invalid].tolist()}, targets={targets_size[invalid].tolist()}"
        )

    from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss

    loss = monotonic_rnnt_loss(
        acts=logits.float().contiguous(),
        labels=targets.contiguous(),
        input_lengths=encoder_states_size.contiguous(),
        label_lengths=targets_size.contiguous(),
        blank_label=model.target_size - 1,
    ).sum()
    run_ctx.mark_as_loss(name="mono_rnnt", loss=loss, inv_norm_factor=loss_norm_factor)
