from typing import Optional, Tuple

from i6_core.returnn import ReturnnTrainingJob

from ...data.librispeech.bpe import bpe_to_vocab_size
from ...data.librispeech.datasets import (
    get_default_bpe_lm_cv_data,
    get_default_bpe_lm_train_data,
)
from ...model_pipelines.common.learning_rates import ConstDecayLRConfig
from ...model_pipelines.common.optimizer import RAdamConfig
from ...model_pipelines.common.train import TrainOptions
from ...model_pipelines.lstm_lm.pytorch_modules import LstmLmConfig
from ...model_pipelines.lstm_lm.train import train


def get_model_config(bpe_size: int = 128) -> LstmLmConfig:
    vocab_size = bpe_to_vocab_size(bpe_size)
    return LstmLmConfig(
        vocab_size=vocab_size,
        embed_dim=512,
        lstm_hidden_size=2048,
        lstm_layers=2,
        dropout=0.2,
    )


def get_train_options(bpe_size: int = 128) -> TrainOptions:
    return TrainOptions(
        train_data_config=get_default_bpe_lm_train_data(bpe_size),
        cv_data_config=get_default_bpe_lm_cv_data(bpe_size),
        save_epochs=[100, 200, 240, 260, 280, 300],
        batch_size=1280,
        accum_grad_multiple_step=1,
        optimizer_config=RAdamConfig(epsilon=1e-08, weight_decay=0, decoupled_weight_decay=False),
        gradient_clip=1.0,
        lr_config=ConstDecayLRConfig(const_lr=1e-03, final_lr=1e-05, const_epochs=100, final_epochs=200),
        num_workers_per_gpu=1,
        automatic_mixed_precision=False,
        gpu_mem_rqmt=11,
        max_seqs=None,
        max_seq_length=None,
    )


def run_training(
    model_config: Optional[LstmLmConfig] = None,
    train_options: Optional[TrainOptions] = None,
) -> Tuple[ReturnnTrainingJob, LstmLmConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    train_job = train(options=train_options, model_config=model_config)

    return train_job, model_config
