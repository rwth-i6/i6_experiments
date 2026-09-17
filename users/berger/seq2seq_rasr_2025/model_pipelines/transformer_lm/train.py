__all__ = ["get_train_step_import"]

import torch
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import Import

from ..common.pytorch_modules import lengths_to_padding_mask
from .pytorch_modules import TransformerLm


def get_train_step_import() -> Import:
    return Import(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        import_as="train_step",
    )


def _train_step(*, model: TransformerLm, data: dict, run_ctx: RunCtx, **_):
    targets = data["data"]  # [B, N]
    targets_size = data["data:size1"]  # [B]
    delayed_targets = data["delayed"]  # [B, N]

    seq_mask = lengths_to_padding_mask(targets_size)

    logits = model.forward(delayed_targets, seq_mask)  # [B, N, V]

    ce_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        targets.long(),
        reduction="none",
    )
    ce_loss = (ce_loss * seq_mask).sum()

    loss_norm_factor = torch.sum(targets_size)

    ppl = torch.exp(ce_loss / loss_norm_factor)

    run_ctx.mark_as_loss(name="ce", loss=ce_loss)
    run_ctx.mark_as_loss(name="ppl", loss=ppl, scale=0)
    run_ctx.mark_as_loss(name="log_ppl", loss=ce_loss, inv_norm_factor=loss_norm_factor, scale=0)
