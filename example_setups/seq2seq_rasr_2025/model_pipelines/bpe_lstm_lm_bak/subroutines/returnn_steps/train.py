import torch
from minireturnn.torch.context import RunCtx

from ...pytorch_modules import LstmLm


def train_step(*, model: LstmLm, data: dict, run_ctx: RunCtx, **_):
    targets = data["data"]  # [B, S]
    targets_size = data["data:size1"]  # [B]

    logits = model.forward(targets, targets_size)

    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        targets, targets_size.cpu(), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    ce_loss = torch.nn.functional.cross_entropy(
        logits.transpose(1, 2),
        targets_masked.long(),
        reduction="sum",
    )  # [B, S]

    loss_norm_factor = torch.sum(targets_size)

    run_ctx.mark_as_loss(name="ce", loss=ce_loss, inv_norm_factor=loss_norm_factor)
    run_ctx.mark_as_loss(name="ppl", loss=torch.exp(ce_loss / loss_norm_factor), scale=0.0)
