__all__ = ["train"]


import torch
from i6_core.returnn.training import ReturnnTrainingJob
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import Import

from ..common.imports import get_model_serializers
from ..common.train import TrainOptions
from ..common.train import train as train_
from .pytorch_modules import ConformerCTCConfig, ConformerCTCModel


def _train_step(*, model: ConformerCTCModel, data: dict, run_ctx: RunCtx, **_):
    audio_samples = data["data"]  # [B, T, 1]
    audio_samples_size = data["data:size1"]  # [B]

    targets = data["classes"].long()  # [B, S]

    targets_size = data["classes:size1"]  # [B]

    log_probs, log_probs_size = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size.to(device=audio_samples.device),
    )  # [B, T, V], [B]

    log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, V]

    loss = torch.nn.functional.ctc_loss(
        log_probs=log_probs,
        targets=targets,
        input_lengths=log_probs_size,
        target_lengths=targets_size,
        blank=model.target_size - 1,
        reduction="sum",
        zero_infinity=True,
    )

    run_ctx.mark_as_loss(name="CTC", loss=loss, inv_norm_factor=torch.sum(targets_size))


def train(options: TrainOptions, model_config: ConformerCTCConfig) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=ConformerCTCModel, model_config=model_config)
    train_step_import = Import(f"{_train_step.__module__}.{_train_step.__name__}", import_as="train_step")

    return train_(options=options, model_serializers=model_serializers, train_step_import=train_step_import)
