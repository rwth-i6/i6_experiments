import warnings
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Optional

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.learning_rate_config import (
    DynamicLearningRateConfig,
    lr_baseline,
    lr_baseline_v3,
    lr_baseline_v2,
    lr_baseline_v4,
    lr_baseline_v5,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.optimizer_config import (
    OptimizerConfig,
    optimizer_baseline,
)


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
    batch_size_factor: int

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


def training_baseline(seed: Optional[int] = None) -> TrainingConfig:
    return TrainingConfig(
        epochs=100,
        partition_epoch_factor=20,
        dynamic_lr=lr_baseline(),
        batch_size=15_000,
        batch_size_factor=160,
        optimizer=optimizer_baseline(),
        num_gpus=1,
        gpu_memory=48,
        torch_amp=TorchAmpTypes.BFLOAT16.value,
        grad_scaler=None,
        random_seed=seed,
    )


def itc_batch_size_80k() -> TrainingConfig:
    return replace(training_baseline(), batch_size=80_000)


def itc_batch_size_150k() -> TrainingConfig:
    return replace(training_baseline(), batch_size=150_000)


def itc_batch_size_250k() -> TrainingConfig:
    return replace(training_baseline(), batch_size=250_000)


def itc_4gpu_setup_v1() -> TrainingConfig:
    """
    Seemed to be failing but was because a faulty node (cn-280)
    :return:
    """
    return replace(
        training_baseline(),
        batch_size=15_000,
        gpu_memory=11,
        num_gpus=4,
        use_torch_amp=False,
        use_grad_scaler=False,
    )


def itc_4gpu_setup_v2() -> TrainingConfig:
    return replace(itc_4gpu_setup_v1(), batch_size=5_000)

def itc_4gpu_setup_v3() -> TrainingConfig:
    return replace(itc_4gpu_setup_v1(), batch_size=13_000)


def bsv2_lrv2() -> TrainingConfig:
    return replace(itc_batch_size_80k(), dynamic_lr=lr_baseline_v2())


def bsv2_lrv3() -> TrainingConfig:
    return replace(itc_batch_size_80k(), dynamic_lr=lr_baseline_v3())


def bsv2_lrv4() -> TrainingConfig:
    return replace(itc_batch_size_80k(), dynamic_lr=lr_baseline_v4())


def bsv2_lrv5() -> TrainingConfig:
    return replace(itc_batch_size_80k(), dynamic_lr=lr_baseline_v5())


def itc_batch_size_80k_150_epochs() -> TrainingConfig:
    return replace(itc_batch_size_80k(), epochs=150)


def itc_batch_size_80k_200_epochs() -> TrainingConfig:
    return replace(itc_batch_size_80k(), epochs=200)


"""
Tests
"""

def training_n2_test() -> TrainingConfig:
    return replace(
        training_baseline(),
        batch_size=3_000,
        gpu_memory=24,
        num_gpus=2,
        use_torch_amp=False,
        use_grad_scaler=False,
        epochs=1,
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
