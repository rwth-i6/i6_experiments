__all__ = ["run", "get_model_config", "get_train_options"]

from typing import Optional

from ....data.librispeech import datasets as librispeech_datasets
from ....model_pipelines.common.learning_rates import OCLRConfig
from ....model_pipelines.common.optimizer import RAdamConfig
from ....model_pipelines.common.train import TrainedModel, train
from ....model_pipelines.ffnn_transducer.pytorch_modules import (
    FFNNTransducerConfig,
    FFNNTransducerModel,
)
from ....model_pipelines.ffnn_transducer.train import FFNNTransducerPrunedTrainOptions, get_pruned_train_step_import
from .ffnn_transducer_bpe import get_model_config


def run(
    descriptor: str,
    model_config: Optional[FFNNTransducerConfig] = None,
    train_options: Optional[FFNNTransducerPrunedTrainOptions] = None,
) -> TrainedModel[FFNNTransducerConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    return train(
        descriptor=descriptor,
        model_class=FFNNTransducerModel,
        model_config=model_config,
        options=train_options,
        train_step_import=get_pruned_train_step_import(train_options),
    )


def get_train_options(bpe_size: int = 128) -> FFNNTransducerPrunedTrainOptions:
    return FFNNTransducerPrunedTrainOptions(
        train_data_config=librispeech_datasets.get_default_bpe_train_data(bpe_size=bpe_size),
        cv_data_config=librispeech_datasets.get_default_bpe_cv_data(bpe_size=bpe_size),
        save_epochs=list(range(1500, 1900, 100)) + list(range(1900, 2001, 20)),
        batch_size=12_000 * 160,
        accum_grad_multiple_step=2,
        optimizer_config=RAdamConfig(
            epsilon=1e-12,
            weight_decay=0.01,
            decoupled_weight_decay=True,
        ),
        lr_config=OCLRConfig(
            init_lr=7e-06,
            peak_lr=5e-04,
            decayed_lr=5e-05,
            final_lr=1e-07,
            inc_epochs=960,
            dec_epochs=960,
            final_epochs=80,
        ),
        gradient_clip=1.0,
        num_workers_per_gpu=2,
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        enc_loss_scale=0.0,
        pred_loss_scale=0.25,
        max_seqs=None,
        max_seq_length=None,
        delay_penalty=0.0,
        skip_epochs_before_pruned_loss=20,
        prune_range=5,
        smoothed_loss_scale=0.5,
    )
