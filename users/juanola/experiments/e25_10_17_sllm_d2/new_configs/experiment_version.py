from enum import Enum

from .experiment_config import exp_v1, exp_v2, ExperimentConfig


class ExperimentVersion(Enum):
    V1 = "v1"
    V2 = "v2"
    # Expand here


_EXPERIMENT_BUILDERS = {
    ExperimentVersion.V1: exp_v1,
    ExperimentVersion.V2: exp_v2,
    # Expand here
}


def get_experiment_config(name: ExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
