from dataclasses import dataclass
from typing import Protocol

from i6_core.returnn import ReturnnConfig


class OptimizerConfig(Protocol):
    def get_returnn_config(self) -> ReturnnConfig: ...


@dataclass
class AdamWConfig:
    epsilon: float
    weight_decay: float

    def get_returnn_config(self) -> ReturnnConfig:
        return ReturnnConfig(
            config={
                "optimizer": {
                    "class": "adamw",
                    "epsilon": self.epsilon,
                    "weight_decay": self.weight_decay,
                },
            },
        )


@dataclass
class RAdamConfig:
    epsilon: float
    weight_decay: float
    decoupled_weight_decay: bool

    def get_returnn_config(self) -> ReturnnConfig:
        return ReturnnConfig(
            config={
                "optimizer": {
                    "class": "radam",
                    "epsilon": self.epsilon,
                    "weight_decay": self.weight_decay,
                    "decoupled_weight_decay": self.decoupled_weight_decay,
                },
            },
        )
