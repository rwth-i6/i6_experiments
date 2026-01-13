from enum import Enum

from .experiment_config import exp_baseline, ExperimentConfig


class LLMExperimentVersion(Enum):
    """
    Avoid duplicate values -> will lead to merged params
    """
    V1_BASELINE = "llm_baseline"


_EXPERIMENT_BUILDERS = {
    LLMExperimentVersion.V1_BASELINE: exp_baseline,
}


def get_experiment_config(name: LLMExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
