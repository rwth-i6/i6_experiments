from sisyphus import tk
from typing import Tuple

from i6_core.returnn import PtCheckpoint
from i6_experiments.common.datasets.librispeech.vocab import get_lm_vocab

from ...data.librispeech.datasets import get_default_word_lm_cv_data, get_default_word_lm_train_data
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


def get_baseline_model_config() -> TransformerLmConfig:
    return TransformerLmConfig(
        vocab_dim=get_lm_vocab(output_prefix="").vocab_size,
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


def get_baseline_train_options() -> TrainOptions:
    return TrainOptions(
        descriptor="baseline",
        train_data_config=get_default_word_lm_train_data(),
        cv_data_config=get_default_word_lm_cv_data(),
        save_epochs=[10, 20, 25, 26, 27, 28, 29, 30],
        batch_size=900,
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
        max_seq_length=602,
    )


def run_word_transformer_lm_baseline(
    num_layers: int = 96,
    prefix: str = "librispeech/word_transformer-lm",
) -> Tuple[TransformerLmConfig, PtCheckpoint]:
    with ExperimentContext(f"{prefix}_{num_layers}l"):
        model_config = get_baseline_model_config()
        model_config.num_layers = num_layers
        train_config = get_baseline_train_options()

        train_job = train(train_config, model_config)
    if num_layers == 96:
        return model_config, tk.Path(
            "/work/asr4/zyang/torch/librispeech/work/i6_core/returnn/training/ReturnnTrainingJob.WuilWP7i1fS2/output/models/epoch.030.pt"
        )
    return model_config, train_job.out_checkpoints[train_config.save_epochs[-1]]  # type: ignore
