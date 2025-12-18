from dataclasses import dataclass, replace
from typing import Dict, Any

from i6_experiments.users.juanola.training.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear


@dataclass(frozen=True)
class DynamicLearningRateConfig:
    """
    LR configuration base dataclass.

    Can contain default values.
    """

    base_lr: float = 1.0
    peak_lr: float = 1e-3
    low_lr: float = 1e-5
    lowest_lr: float = 1e-6

    step_peak_fraction: float = 0.45
    step_finetune_fraction: float = 0.9

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        assert (
            self.step_peak_fraction <= self.step_finetune_fraction
        ), f"step_peak_fraction ({self.step_peak_fraction}) should be before step_finetune_fraction ({self.step_finetune_fraction})"

        assert (
            self.lowest_lr <= self.low_lr
        ), f"low_lr ({self.low_lr}) should not be lower than lowest_lr ({self.lowest_lr})"
        assert self.low_lr <= self.peak_lr, f"peak_lr ({self.peak_lr}) should not be lower than low_lr ({self.low_lr})"

    def get_dynamic_lr_returnn_config(self, train_epochs: int) -> Dict[str, Any]:
        """
        Contains Returnn logic.

        # TODO: maybe this logic should be in the experiments
        :param train_epochs:
        :return:
        """
        return {
            "learning_rate": self.base_lr,
            "dynamic_learning_rate": dyn_lr_piecewise_linear,
            "learning_rate_piecewise_by_epoch_continuous": True,
            "learning_rate_piecewise_steps": [
                self.step_peak_fraction * train_epochs,
                self.step_finetune_fraction * train_epochs,
                train_epochs,
            ],
            "learning_rate_piecewise_values": [self.low_lr, self.peak_lr, self.low_lr, self.lowest_lr],
        }


"""
Specific configurations set below.
"""


def lr_baseline() -> DynamicLearningRateConfig:
    return DynamicLearningRateConfig()


def lr_baseline_v2() -> DynamicLearningRateConfig:
    return replace(lr_baseline(), peak_lr=1e-4)


def lr_baseline_v3() -> DynamicLearningRateConfig:
    return replace(lr_baseline(), peak_lr=1e-5)


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
