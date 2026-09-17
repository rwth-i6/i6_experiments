__all__ = ["CTCMultiOutputTrainOptions", "get_train_step_import"]


from dataclasses import dataclass
from typing import List

import torch
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import PartialImport

from ..common.train import TrainOptions
from .pytorch_modules import ConformerCTCMultiOutputModel


@dataclass
class CTCMultiOutputTrainOptions(TrainOptions):
    target_names: List[str]


def get_train_step_import(options: CTCMultiOutputTrainOptions) -> PartialImport:
    return PartialImport(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        hashed_arguments={"target_names": options.target_names},
        unhashed_arguments={},
        unhashed_package_root="",
        import_as="train_step",
    )


def _train_step(*, model: ConformerCTCMultiOutputModel, data: dict, run_ctx: RunCtx, target_names: List[str], **_):
    audio_samples = data["data"]  # [B, T, 1]
    audio_samples_size = data["data:size1"]  # [B]

    log_probs_list, log_probs_size = model.forward(
        audio_samples=audio_samples,
        audio_samples_size=audio_samples_size.to(device=audio_samples.device),
    )  # [B, T, V], [B]

    for log_probs, target_name in zip(log_probs_list, target_names):
        targets = data[target_name].long()  # [B, S]
        targets_size = data[f"{target_name}:size1"]  # [B]

        log_probs = torch.transpose(log_probs, 0, 1)  # [T, B, V]

        loss = torch.nn.functional.ctc_loss(
            log_probs=log_probs,
            targets=targets,
            input_lengths=log_probs_size,
            target_lengths=targets_size,
            blank=log_probs.size(2) - 1,
            reduction="sum",
            zero_infinity=True,
        )

        run_ctx.mark_as_loss(name=f"CTC_{target_name}", loss=loss, inv_norm_factor=torch.sum(targets_size))
