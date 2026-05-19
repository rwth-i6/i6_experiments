__all__ = ["run", "get_model_config", "get_train_options"]

from typing import Optional

from .....data.loquacious.datasets import get_default_word_lm_train_data, get_lm_vocab, get_medium_word_lm_cv_data
from .....model_pipelines.common.learning_rates import NewbobRelConfig
from .....model_pipelines.common.optimizer import SGDConfig
from .....model_pipelines.common.train import TrainedModel, TrainOptions, train
from .....model_pipelines.transformer_lm.pytorch_modules import (
    PositionalEncodingConfig,
    TransformerBlockConfig,
    TransformerLinearConfig,
    TransformerLm,
    TransformerLmConfig,
    TransformerMHSAConfig,
)
from .....model_pipelines.transformer_lm.train import get_train_step_import


def run(
    descriptor: str,
    model_config: Optional[TransformerLmConfig] = None,
    train_options: Optional[TrainOptions] = None,
) -> TrainedModel[TransformerLmConfig]:
    if model_config is None:
        model_config = get_model_config()
    if train_options is None:
        train_options = get_train_options()

    return train(
        descriptor=descriptor,
        model_class=TransformerLm,
        model_config=model_config,
        options=train_options,
        train_step_import=get_train_step_import(),
    )


def get_model_config() -> TransformerLmConfig:
    return TransformerLmConfig(
        vocab_dim=get_lm_vocab().vocab_size,
        embed_dim=128,
        hid_dim=512,
        num_layers=96,
        block_config=TransformerBlockConfig(
            linear_config=TransformerLinearConfig(input_dim=512, ff_dim=2048, output_dim=512, dropout=0.0),
            mhsa_config=TransformerMHSAConfig(
                input_dim=512,
                num_heads=8,
                dropout=0.0,
            ),
        ),
        pos_enc_config=PositionalEncodingConfig(
            embed_dim=128,
            max_len=5000,
            dropout=0.0,
        ),
        dropout=0.0,
    )


def get_train_options() -> TrainOptions:
    return TrainOptions(
        train_data_config=get_default_word_lm_train_data(),
        cv_data_config=get_medium_word_lm_cv_data(),
        save_epochs=[10, 20, 25, 26, 27, 28, 29, 30],
        batch_size=900,
        optimizer_config=SGDConfig(weight_decay=0.0),
        lr_config=NewbobRelConfig(
            learning_rate=1.0,
            lr_decay=0.9,
            error_measure="ppl",
            multi_num_epochs=10,
            multi_update_interval=1,
            relative_error_div_by_old=True,
            relative_error_threshold=-0.005,
        ),
        num_workers_per_gpu=1,
        max_seqs=64,
        max_seq_length=602,
    )
