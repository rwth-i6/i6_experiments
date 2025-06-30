__all__ = ["AEDTrainOptions", "train"]

from dataclasses import dataclass
import torch
from i6_core.returnn.training import ReturnnTrainingJob
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import PartialImport

from ..common.imports import get_model_serializers
from ..common.train import TrainOptions
from ..common.train import train as train_
from .pytorch_modules import AEDConfig, AEDModel


@dataclass
class AEDTrainOptions(TrainOptions):
    ctc_loss_scale: float
    label_smoothing: float
    label_smoothing_start_epoch: int


def _train_step(
    *,
    model: AEDModel,
    data: dict,
    run_ctx: RunCtx,
    ctc_loss_scale: float = 0.0,
    label_smoothing: float = 0.0,
    label_smoothing_start_epoch: int = 1,
    **_,
):
    audio_samples = data["data"]  # [B, T, 1]
    audio_samples_size = data["data:size1"]  # [B]

    labels = data["classes"]  # [B, S]
    labels_len = data["classes:size1"]  # [B]

    decoder_logits, audio_features_len, ctc_logprobs = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size,
        bpe_labels=labels,
    )

    if ctc_loss_scale != 0:
        transposed_logprobs = torch.permute(ctc_logprobs, (1, 0, 2))  # CTC needs [T, B, F]
        ctc_loss = torch.nn.functional.ctc_loss(
            transposed_logprobs,
            labels,
            input_lengths=audio_features_len,
            target_lengths=labels_len,
            blank=model.label_target_size,
            reduction="sum",
            zero_infinity=True,
        )

        num_phonemes = torch.sum(labels_len)
        run_ctx.mark_as_loss(
            name="ctc",
            loss=ctc_loss,
            scale=ctc_loss_scale,
            inv_norm_factor=num_phonemes,
        )

    # ignore padded values in the loss
    targets_packed = torch.nn.utils.rnn.pack_padded_sequence(
        labels, labels_len.cpu(), batch_first=True, enforce_sorted=False
    )
    targets_masked, _ = torch.nn.utils.rnn.pad_packed_sequence(targets_packed, batch_first=True, padding_value=-100)

    label_smoothing = label_smoothing if run_ctx.epoch >= label_smoothing_start_epoch else 0.0

    ce_loss = torch.nn.functional.cross_entropy(
        decoder_logits.transpose(1, 2),
        targets_masked.long(),
        reduction="sum",
        label_smoothing=label_smoothing,
    )  # [B,N]

    num_labels = torch.sum(labels_len)
    run_ctx.mark_as_loss(name="decoder_ce", loss=ce_loss, inv_norm_factor=num_labels)


def train(
    options: AEDTrainOptions,
    model_config: AEDConfig,
) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=AEDModel, model_config=model_config)
    train_step_import = PartialImport(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        hashed_arguments={
            "ctc_loss_scale": options.ctc_loss_scale,
            "label_smoothing": options.label_smoothing,
            "label_smoothing_start_epoch": options.label_smoothing_start_epoch,
        },
        unhashed_arguments={},
        unhashed_package_root="",
        import_as="train_step",
    )

    return train_(options=options, model_serializers=model_serializers, train_step_import=train_step_import)
