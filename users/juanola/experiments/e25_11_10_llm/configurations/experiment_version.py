from enum import Enum

from .experiment_config import exp_baseline, ExperimentConfig, exp_baseline_test


class LLMExperimentVersion(Enum):
    """
    Avoid duplicate values -> will lead to merged params
    """
    V1_BASELINE = "llm_baseline"

    # Test
    V1_BASELINE_TEST = "llm_baseline_test"


_EXPERIMENT_BUILDERS = {
    LLMExperimentVersion.V1_BASELINE: exp_baseline,
    LLMExperimentVersion.V1_BASELINE_TEST: exp_baseline_test,
}


def get_experiment_config(name: LLMExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
