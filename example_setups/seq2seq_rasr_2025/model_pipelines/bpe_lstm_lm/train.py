__all__ = ["train"]

import torch
from i6_core.returnn.training import ReturnnTrainingJob
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import Import

from ..common.imports import get_model_serializers
from ..common.pytorch_modules import lengths_to_padding_mask
from ..common.train import TrainOptions
from ..common.train import train as train_
from .pytorch_modules import LstmLm, LstmLmConfig


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


def train(
    options: TrainOptions,
    model_config: LstmLmConfig,
) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=LstmLm, model_config=model_config)
    train_step_import = Import(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        import_as="train_step",
    )

    return train_(options=options, model_serializers=model_serializers, train_step_import=train_step_import)
