import inspect
from i6_core import returnn
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, TYPE_CHECKING, List

from sisyphus.hash import sis_hash_helper

if TYPE_CHECKING:
    import optuna


@dataclass
class OptunaReturnnConfig:
    config_generator: Callable[..., returnn.ReturnnConfig]
    config_kwargs: Dict[str, Any]
    config_updates: List[returnn.ReturnnConfig] = field(default_factory=list)

    def generate_config(self, trial: "optuna.Trial") -> returnn.ReturnnConfig:
        returnn_config = self.config_generator(trial, **self.config_kwargs)
        for update_config in self.config_updates:
            returnn_config.update(update_config)

        return returnn_config

    def update(self, other: returnn.ReturnnConfig) -> None:
        self.config_updates.append(other)

    def _sis_hash(self):
        return sis_hash_helper(
            {
                "config_generator": inspect.getsource(self.config_generator),
                "config_kwargs": self.config_kwargs,
                "config_updates": self.config_updates,
            }
        )
