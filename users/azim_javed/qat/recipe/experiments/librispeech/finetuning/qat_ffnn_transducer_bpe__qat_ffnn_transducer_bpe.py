__all__ = ["run", "get_model_config", "get_train_options"]

from typing import Optional

from ....model_pipelines.common.train import FinetuneOptions, TrainedModel
from ....model_pipelines.qat_ffnn_transducer.pytorch_modules import QATFFNNTransducerConfig
from ....model_pipelines.qat_ffnn_transducer.train import QATFFNNTransducerTrainOptions
from .qat_ffnn_transducer_bpe__ffnn_transducer_bpe import get_model_config, get_train_options
from .qat_ffnn_transducer_bpe__ffnn_transducer_bpe import run as run_finetuning


def run(
    descriptor: str,
    base_model: TrainedModel[QATFFNNTransducerConfig],
    qat_args: Optional[dict] = None,
    model_config: Optional[QATFFNNTransducerConfig] = None,
    train_options: Optional[QATFFNNTransducerTrainOptions] = None,
) -> TrainedModel[QATFFNNTransducerConfig]:
    return run_finetuning(
        descriptor=descriptor,
        base_model=base_model,
        qat_args=qat_args,
        model_config=model_config,
        train_options=train_options,
        finetune_options=FinetuneOptions(
            label="full_qat_model",
            filename=base_model.get_checkpoint(),
            init_for_train=True,
            ignore_missing=False,
        ),
    )
