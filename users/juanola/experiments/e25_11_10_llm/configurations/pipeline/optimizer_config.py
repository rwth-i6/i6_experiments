from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OptimizerConfig:
    """
    Optimizer configuration base dataclass.

    Can contain default values.
    """

    epsilon: float = 1e-16
    weight_decay: float = 0.01

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


    def get_optimizer_returnn_config(self) -> dict[str, dict[str, Any]]:
        """
        Contains Returnn logic.

        # TODO: maybe this logic should be in the experiments
        :return:
        """
        return {
            "optimizer": {
                "class": "adamw",
                "epsilon": self.epsilon,
                "weight_decay": self.weight_decay,
            },
        }


"""
Specific configurations set below.
"""


def optimizer_baseline() -> OptimizerConfig:
    return OptimizerConfig()


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
