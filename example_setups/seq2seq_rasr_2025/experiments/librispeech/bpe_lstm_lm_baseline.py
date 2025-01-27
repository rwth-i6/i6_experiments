from typing import Tuple

from i6_core.returnn import PtCheckpoint

from ...data.librispeech.bpe import bpe_to_vocab_size
from ...data.librispeech.datasets import (
    get_default_bpe_lm_cv_data,
    get_default_bpe_lm_train_data,
)
from ...model_pipelines.bpe_lstm_lm.pytorch_modules import LstmLmConfig
from ...model_pipelines.bpe_lstm_lm.train import train
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.learning_rates import ConstDecayLRConfig
from ...model_pipelines.common.optimizer import RAdamConfig
from ...model_pipelines.common.train import TrainOptions

BPE_SIZE = 128


def get_baseline_model_config() -> LstmLmConfig:
    vocab_size = bpe_to_vocab_size(BPE_SIZE)
    return LstmLmConfig(
        vocab_size=vocab_size,
        embed_dim=512,
        lstm_hidden_size=2048,
        lstm_layers=2,
        dropout=0.2,
    )


def get_baseline_train_options() -> TrainOptions:
    return TrainOptions(
        descriptor="baseline",
        train_data_config=get_default_bpe_lm_train_data(BPE_SIZE),
        cv_data_config=get_default_bpe_lm_cv_data(BPE_SIZE),
        save_epochs=[100, 200, 240, 260, 280, 300],
        batch_size=1280,
        accum_grad_multiple_step=1,
        optimizer_config=RAdamConfig(epsilon=1e-08, weight_decay=0, decoupled_weight_decay=False),
        gradient_clip=1.0,
        lr_config=ConstDecayLRConfig(const_lr=1e-03, final_lr=1e-05, const_epochs=100, final_epochs=200),
        num_workers_per_gpu=1,
        stop_on_inf_nan_score=False,
    )


def run_bpe_lstm_lm_baseline(prefix: str = "librispeech/bpe_lstm-lm") -> Tuple[LstmLmConfig, PtCheckpoint]:
    with ExperimentContext(prefix):
        model_config = get_baseline_model_config()
        train_config = get_baseline_train_options()

        train_job = train(train_config, model_config)
    return (model_config, train_job.out_checkpoints[200])  # type: ignore
