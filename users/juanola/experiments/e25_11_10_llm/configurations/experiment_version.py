from enum import Enum

from .experiment_config import ExperimentConfig, exp_baseline_test, exp_1_1, exp_1_2, exp_1_3, exp_1_4


class LLMExperimentVersion(Enum):
    """
    Avoid duplicate values -> will lead to merged params
    """
    V1_1_BASE_TRANS_DATA = "llm_base_trans_data"
    V1_2_BASE_COMB_DATA = "llm_base_comb_data"
    V1_3_SMALL_TRANS_DATA = "llm_small_trans_data"
    V1_4_SMALL_COMB_DATA = "llm_small_comb_data"

    # Tests
    V1_BASELINE_TEST = "llm_baseline_test"


_EXPERIMENT_BUILDERS = {
    LLMExperimentVersion.V1_1_BASE_TRANS_DATA: exp_1_1,
    LLMExperimentVersion.V1_2_BASE_COMB_DATA: exp_1_2,
    LLMExperimentVersion.V1_3_SMALL_TRANS_DATA: exp_1_3,
    LLMExperimentVersion.V1_4_SMALL_COMB_DATA: exp_1_4,

    # Tests
    LLMExperimentVersion.V1_BASELINE_TEST: exp_baseline_test,
}


def get_experiment_config(name: LLMExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
