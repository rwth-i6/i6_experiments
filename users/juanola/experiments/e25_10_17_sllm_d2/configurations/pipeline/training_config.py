from dataclasses import dataclass, replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.learning_rate_config import \
    DynamicLearningRateConfig, lr_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.optimizer_config import \
    OptimizerConfig, optimizer_baseline


@dataclass(frozen=True)
class TrainingConfig:
    """
    Training configuration base dataclass.

    Can contain default values.
    """
    epochs: int
    partition_epoch_factor: int

    dynamic_lr: DynamicLearningRateConfig

    optimizer: OptimizerConfig

    batch_size: int
    batch_size_factor: int

    num_gpus: int # Should be 1 for 48gb in i6 cluster
    gpu_memory: int


"""
Specific configurations set below.
"""


def training_baseline() -> TrainingConfig:
    return TrainingConfig(
        epochs=100,
        partition_epoch_factor=20,

        dynamic_lr=lr_baseline(),

        batch_size=15_000,
        batch_size_factor=160,

        optimizer=optimizer_baseline(),

        num_gpus=1,
        gpu_memory=48,
    )


def training_v2() -> TrainingConfig:
    return replace(training_baseline(), batch_size=30_000)

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
