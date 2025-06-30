__all__ = ["train"]


from dataclasses import dataclass
from typing import List
import torch
from i6_core.returnn.training import ReturnnTrainingJob
from minireturnn.torch.context import RunCtx

from i6_experiments.common.setups.serialization import PartialImport

from ..common.serializers import get_model_serializers
from ..common.train import TrainOptions
from ..common.train import train as train_
from .pytorch_modules import ConformerCTCMultiOutputConfig, ConformerCTCMultiOutputModel


@dataclass
class CTCMultiOutputTrainOptions(TrainOptions):
    target_names: List[str]


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


def train(options: CTCMultiOutputTrainOptions, model_config: ConformerCTCMultiOutputConfig) -> ReturnnTrainingJob:
    model_serializers = get_model_serializers(model_class=ConformerCTCMultiOutputModel, model_config=model_config)
    train_step_import = PartialImport(
        code_object_path=f"{_train_step.__module__}.{_train_step.__name__}",
        hashed_arguments={"target_names": options.target_names},
        unhashed_arguments={},
        unhashed_package_root="",
        import_as="train_step",
    )

    return train_(options=options, model_serializers=model_serializers, train_step_import=train_step_import)
