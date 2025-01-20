from ...data.switchboard.bpe import bpe_to_vocab_size
from ...data.switchboard.datasets import (
    get_default_bpe_lm_cv_data,
    get_default_bpe_lm_train_data,
)
from ...model_pipelines.bpe_lstm_lm.pipeline import PipelineConfig, run_pipeline
from ...model_pipelines.bpe_lstm_lm.pytorch_modules import LstmLmConfig
from ...model_pipelines.bpe_lstm_lm.subroutines.configs import (
    ConstDecayLRConfig,
    TrainRoutineConfig,
)


def get_baseline_config(bpe_size: int) -> PipelineConfig:
    vocab_size = bpe_to_vocab_size(bpe_size)
    model_config = LstmLmConfig(
        vocab_size=vocab_size,
        target_embed_dim=512,
        lstm_hidden_size=2048,
        lstm_layers=2,
        dropout=0.2,
        final_dropout=0.2,
    )

    train_routine_config = TrainRoutineConfig(
        train_data_config=get_default_bpe_lm_train_data(bpe_size),
        cv_data_config=get_default_bpe_lm_cv_data(bpe_size),
        num_epochs=300,
        batch_size=1280,
        lr_config=ConstDecayLRConfig(
            const_lr=1e-03,
            final_lr=1e-05,
            const_epochs=200,
            final_epochs=100,
        ),
        gradient_clip=1.0,
    )

    return PipelineConfig(
        model_config=model_config,
        train_config=train_routine_config,
    )


def run_bpe_lstm_lm_baselines() -> None:
    for bpe_size in [128, 5000]:
        run_pipeline(get_baseline_config(bpe_size), name=f"bpe-{bpe_size}", prefix="switchboard")
