__all__ = ["ConstConstDecayLRConfig", "ConstDecayLRConfig", "OCLRConfig"]

from dataclasses import dataclass
from typing import Protocol

from i6_core.returnn.config import CodeWrapper, ReturnnConfig


class LRConfig(Protocol):
    def get_returnn_config(self) -> ReturnnConfig: ...


@dataclass
class ConstConstDecayLRConfig:
    const_lr_1: float
    const_lr_2: float
    decayed_lr: float
    final_lr: float
    const_epochs_1: int
    const_epochs_2: int
    dec_epochs: int
    final_epochs: int

    def get_returnn_config(self) -> ReturnnConfig:
        return ReturnnConfig(
            config={
                "learning_rates": CodeWrapper(
                    f"[{self.const_lr_1}] * {self.const_epochs_1}"
                    f"+ [{self.const_lr_2}] * {self.const_epochs_2}"
                    f"+ list(np.linspace({self.const_lr_2}, {self.decayed_lr}, {self.dec_epochs}))"
                    f"+ list(np.linspace({self.decayed_lr}, {self.final_lr}, {self.final_epochs}))"
                )
            },
            python_prolog=["import numpy as np"],
        )


@dataclass
class ConstDecayLRConfig:
    const_lr: float
    final_lr: float
    const_epochs: int
    final_epochs: int

    def get_returnn_config(self) -> ReturnnConfig:
        return ReturnnConfig(
            python_prolog=["import numpy as np"],
            config={
                "learning_rates": CodeWrapper(
                    f"([{self.const_lr}] * {self.const_epochs}) + list(np.linspace({self.const_lr}, {self.final_lr}, {self.final_epochs}))"
                ),
            },
        )


@dataclass
class OCLRConfig:
    init_lr: float
    peak_lr: float
    decayed_lr: float
    final_lr: float
    inc_epochs: int
    dec_epochs: int
    final_epochs: int

    def get_returnn_config(self) -> ReturnnConfig:
        return ReturnnConfig(
            config={
                "learning_rates": CodeWrapper(
                    f"list(np.linspace({self.init_lr}, {self.peak_lr}, {self.inc_epochs}))"
                    f"+ list(np.linspace({self.peak_lr}, {self.decayed_lr}, {self.dec_epochs}))"
                    f"+ list(np.linspace({self.decayed_lr}, {self.final_lr}, {self.final_epochs}))"
                )
            },
            python_prolog=["import numpy as np"],
        )
