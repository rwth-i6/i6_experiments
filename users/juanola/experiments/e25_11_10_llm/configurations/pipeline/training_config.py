import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .learning_rate_config import DynamicLearningRateConfig, lr_baseline
from .optimizer_config import OptimizerConfig, optimizer_baseline


class TorchAmpTypes(Enum):
    BFLOAT16 = "bfloat16"
    FLOAT16 = "float16"


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

    num_gpus: int  # Should be 1 for 48gb in i6 cluster
    gpu_memory: int

    torch_amp: str = None
    grad_scaler: Any = None

    use_torch_amp: bool = True
    use_grad_scaler: bool = True

    max_seq_length_seconds: int = 19.5

    random_seed: Optional[int] = None  # Added to config only if specified

    debug_returnn_param: bool = True

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        if self.use_torch_amp:
            assert self.torch_amp is not None, f"If torch_amp is used it should not be None ({self.torch_amp})"

            if self.gpu_memory == 11:
                assert (
                        self.torch_amp != TorchAmpTypes.BFLOAT16.value
                ), "torch_amp with 11Gb nodes should not be bfloat16."


"""
Specific configurations set below.
"""


def training_transcript_data(seed: Optional[int] = None) -> TrainingConfig:
    return TrainingConfig(
        epochs=100,
        partition_epoch_factor=1,  # !!!
        dynamic_lr=lr_baseline(),
        batch_size=15_000,  # !!!
        optimizer=optimizer_baseline(),
        num_gpus=1,
        gpu_memory=48,
        torch_amp=TorchAmpTypes.BFLOAT16.value,
        grad_scaler=None,
        random_seed=seed,
    )


def training_lm_data() -> TrainingConfig:
    return dataclasses.replace(training_transcript_data(),
                               epochs=5,
                               partition_epoch_factor=20,
                               )


"""
Test
"""

def training_baseline_test() -> TrainingConfig:
    return dataclasses.replace(training_transcript_data(),
                               epochs=1,
                               batch_size=1_000,
                               gpu_memory=48)

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
