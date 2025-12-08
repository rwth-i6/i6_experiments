from dataclasses import dataclass, replace


@dataclass(frozen=True)
class TrainingConfig:
    """
    Training configuration base dataclass.

    Can contain default values.
    """
    epochs: int
    partition_epoch_factor: int

    batch_size: int

    num_gpus: int
    gpu_memory: int


"""
Specific configurations set below.
"""


def training_baseline() -> TrainingConfig:
    return TrainingConfig(
        epochs=100,
        partition_epoch_factor=20,

        batch_size=15_000,

        num_gpus=1,
        gpu_memory=48,
    )


def training_v2() -> TrainingConfig:
    return replace(training_baseline(), batch_size=30_000)

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
