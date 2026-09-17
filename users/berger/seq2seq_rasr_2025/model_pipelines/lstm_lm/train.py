__all__ = ["get_train_step_import"]

import torch
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import Import

from ..common.pytorch_modules import lengths_to_padding_mask
from .pytorch_modules import LstmLm


def get_train_step_import() -> Import:
    return Import(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        import_as="train_step",
    )


def _train_step(*, model: LstmLm, data: dict, run_ctx: RunCtx, **_):
    targets = data["data"]  # [B, S]
    targets_size = data["data:size1"]  # [B]
    delayed_targets = data["delayed"]  # [B, S]

    logits = model.forward(delayed_targets)

    ce_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        targets.long(),
        reduction="none",
    )
    seq_mask = lengths_to_padding_mask(targets_size)
    ce_loss = (ce_loss * seq_mask).sum()

    loss_norm_factor = torch.sum(targets_size)

    run_ctx.mark_as_loss(name="ce", loss=ce_loss, inv_norm_factor=loss_norm_factor)
