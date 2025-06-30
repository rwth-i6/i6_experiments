from typing import Tuple

from ...data.librispeech.bpe import bpe_to_vocab_size
from i6_core.returnn import PtCheckpoint

from ...data.librispeech.datasets import get_default_bpe_lm_cv_data, get_default_bpe_lm_train_data
from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.learning_rates import NewbobRelConfig
from ...model_pipelines.common.optimizer import SGDConfig
from ...model_pipelines.common.train import TrainOptions
from ...model_pipelines.transformer_lm.pytorch_modules import (
    PositionalEncodingConfig,
    TransformerBlockConfig,
    TransformerLinearConfig,
    TransformerLmConfig,
    TransformerMHSAConfig,
)
from ...model_pipelines.transformer_lm.train import train

BPE_SIZE = 128


def get_baseline_model_config(bpe_size: int) -> TransformerLmConfig:
    return TransformerLmConfig(
        vocab_dim=bpe_to_vocab_size(bpe_size),
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


def get_baseline_train_options(bpe_size: int) -> TrainOptions:
    if bpe_size == 128:
        # BPE 128: average 2.6033 BPE per word
        max_seq_length = 1560
        batch_size = 2340
    elif bpe_size == 5000:
        # BPE 5000: average 1.2554 BPE per word
        max_seq_length = 750
        batch_size = 1125
    else:
        raise NotImplementedError

    return TrainOptions(
        descriptor="baseline",
        train_data_config=get_default_bpe_lm_train_data(bpe_size),
        cv_data_config=get_default_bpe_lm_cv_data(bpe_size),
        save_epochs=[10, 20, 25, 26, 27, 28, 29, 30],
        batch_size=batch_size,
        accum_grad_multiple_step=1,
        optimizer_config=SGDConfig(weight_decay=0.0),
        gradient_clip=1.0,
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
        automatic_mixed_precision=True,
        gpu_mem_rqmt=24,
        max_seqs=64,
        max_seq_length=max_seq_length,
    )


def run_bpe_transformer_lm_baseline(
    bpe_size: int,
    num_layers: int = 96,
    prefix: str = "librispeech/bpe_transformer-lm",
) -> Tuple[TransformerLmConfig, PtCheckpoint]:
    with ExperimentContext(f"{prefix}_{num_layers}l_bpe-{bpe_size}"):
        model_config = get_baseline_model_config(bpe_size)
        model_config.num_layers = num_layers
        train_config = get_baseline_train_options(bpe_size)

        train_job = train(train_config, model_config)
    return model_config, train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore
