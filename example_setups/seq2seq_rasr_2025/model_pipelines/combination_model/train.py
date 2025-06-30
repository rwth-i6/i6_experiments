__all__ = ["CombinationTrainOptions", "train"]

from dataclasses import dataclass
import torch
from i6_core.returnn.training import ReturnnTrainingJob
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import PartialImport

from ..common.imports import get_model_serializers
from ..common.train import TrainOptions
from ..common.train import train as train_
from .pytorch_modules import CombinationModelConfig, CombinationModel


@dataclass
class CombinationTrainOptions(TrainOptions):
    ctc_loss_scale: float
    transducer_loss_scale: float
    attention_loss_scale: float
    attention_label_smoothing: float
    attention_label_smoothing_start_epoch: int


def _train_step(
    *,
    model: CombinationModel,
    data: dict,
    run_ctx: RunCtx,
    ctc_loss_scale: float = 0.0,
    transducer_loss_scale: float = 0.0,
    attention_loss_scale: float = 0.0,
    attention_label_smoothing: float = 0.0,
    attention_label_smoothing_start_epoch: int = 1,
    **_,
):
    audio_samples = data["data"]  # [B, T, 1]
    audio_samples_size = data["data:size1"]  # [B]

    targets = data["classes"]  # [B, S]
    targets_size = data["classes:size1"]  # [B]
    total_targets = torch.sum(targets_size)

    transducer_logits, attention_logits, ctc_log_probs, encoder_states_size = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
        targets=targets,
        targets_size=targets_size,
    )

    if ctc_loss_scale != 0:
        transposed_logprobs = torch.permute(ctc_log_probs, (1, 0, 2))  # CTC needs [T, B, F]
        ctc_loss = torch.nn.functional.ctc_loss(
            transposed_logprobs,
            targets,
            input_lengths=encoder_states_size,
            target_lengths=targets_size,
            blank=model.target_size - 1,
            reduction="sum",
            zero_infinity=True,
        )

        run_ctx.mark_as_loss(
            name="ctc",
            loss=ctc_loss,
            scale=ctc_loss_scale,
            inv_norm_factor=total_targets,
        )

    if transducer_loss_scale != 0:
        from i6_native_ops.monotonic_rnnt import monotonic_rnnt_loss

        rnnt_loss = monotonic_rnnt_loss(
            acts=transducer_logits.to(dtype=torch.float32),
            labels=targets,
            input_lengths=encoder_states_size,
            label_lengths=targets_size,
            blank_label=model.target_size - 1,
        ).sum()

        run_ctx.mark_as_loss(
            name="mono_rnnt",
            loss=rnnt_loss,
            scale=transducer_loss_scale,
            inv_norm_factor=total_targets,
        )

    if attention_loss_scale != 0:
        # ignore padded values in the loss
        targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
            targets, targets_size.cpu(), batch_first=True, enforce_sorted=False
        )
        targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

        label_smoothing = attention_label_smoothing if run_ctx.epoch >= attention_label_smoothing_start_epoch else 0.0

        ce_loss = torch.nn.functional.cross_entropy(
            attention_logits.transpose(1, 2),
            targets_masked.long(),
            reduction="sum",
            label_smoothing=label_smoothing,
        )  # [B,N]

        run_ctx.mark_as_loss(
            name="attention_ce", loss=ce_loss, scale=attention_loss_scale, inv_norm_factor=total_targets
        )


def train(
    options: CombinationTrainOptions,
    model_config: CombinationModelConfig,
) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=CombinationModel, model_config=model_config)
    train_step_import = PartialImport(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        hashed_arguments={
            "ctc_loss_scale": options.ctc_loss_scale,
            "transducer_loss_scale": options.transducer_loss_scale,
            "attention_loss_scale": options.attention_loss_scale,
            "attention_label_smoothing": options.attention_label_smoothing,
            "attention_label_smoothing_start_epoch": options.attention_label_smoothing_start_epoch,
        },
        unhashed_arguments={},
        unhashed_package_root="",
        import_as="train_step",
    )

    return train_(options=options, model_serializers=model_serializers, train_step_import=train_step_import)
