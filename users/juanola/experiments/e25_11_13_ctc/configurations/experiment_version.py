from enum import Enum

from .experiment_config import exp_baseline, ExperimentConfig


class CTCExperimentVersion(Enum):
    """
    Avoid duplicate values -> will lead to merged params
    """
    V1_BASELINE = "baseline"



_EXPERIMENT_BUILDERS = {
    CTCExperimentVersion.V1_BASELINE: exp_baseline,
}


def get_experiment_config(name: CTCExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
