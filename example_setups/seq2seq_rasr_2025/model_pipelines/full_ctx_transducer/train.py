__all__ = ["LstmTransducerTrainOptions", "train"]

from dataclasses import dataclass

import torch
from i6_core.returnn.training import ReturnnTrainingJob
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import PartialImport

from ..common.imports import get_model_serializers
from ..common.train import TrainOptions
from ..common.train import train as train_
from .pytorch_modules import LstmTransducerConfig, LstmTransducerModel


@dataclass
class LstmTransducerTrainOptions(TrainOptions):
    ctc_loss_scale: float


def _train_step(
    *,
    model: LstmTransducerModel,
    data: dict,
    run_ctx: RunCtx,
    ctc_loss_scale: float = 0.0,
    **_,
):
    from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss

    audio_samples = data["data"]  # [B, T, 1]
    audio_samples_size = data["data:size1"].to(device=audio_samples.device)  # [B]

    targets = data["classes"]  # [B, S]
    targets_size = data["classes:size1"]  # [B]
    targets_size = targets_size.to(device=audio_samples.device)

    logits, ctc_log_probs, encoder_states_size = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
        targets=targets,
        targets_size=targets_size,
    )

    rnnt_loss = monotonic_rnnt_loss(
        acts=logits.to(dtype=torch.float32),
        labels=targets,
        input_lengths=encoder_states_size,
        label_lengths=targets_size,
        blank_label=model.target_size - 1,
    ).sum()

    loss_norm_factor = torch.sum(targets_size)

    run_ctx.mark_as_loss(name="mono_rnnt", loss=rnnt_loss, inv_norm_factor=loss_norm_factor)

    if ctc_loss_scale != 0:
        ctc_log_probs = torch.transpose(ctc_log_probs, 0, 1)  # [T, B, C]

        loss = torch.nn.functional.ctc_loss(
            log_probs=ctc_log_probs,
            targets=targets,
            input_lengths=encoder_states_size,
            target_lengths=targets_size,
            blank=model.target_size - 1,
            reduction="sum",
            zero_infinity=True,
        )

        run_ctx.mark_as_loss(
            name="ctc",
            loss=loss,
            scale=ctc_loss_scale,
            inv_norm_factor=loss_norm_factor,
        )


def train(
    options: LstmTransducerTrainOptions,
    model_config: LstmTransducerConfig,
) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=LstmTransducerModel, model_config=model_config)
    train_step_import = PartialImport(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        hashed_arguments={"ctc_loss_scale": options.ctc_loss_scale},
        unhashed_arguments={},
        unhashed_package_root="",
        import_as="train_step",
    )

    return train_(options=options, model_serializers=model_serializers, train_step_import=train_step_import)
