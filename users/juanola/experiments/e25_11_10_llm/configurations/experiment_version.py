from enum import Enum

from .experiment_config import ExperimentConfig, exp_baseline_test, exp_1_1, exp_1_2, exp_1_3, exp_1_4, exp_1_5, \
    exp_1_6, base_comb_sllm_vocab, small_comb_sllm_vocab, base_llm_sllm_vocab, small_llm_sllm_vocab


class LLMExperimentVersion(Enum):
    """
    Avoid duplicate values -> will lead to merged params
    """
    V1_1_BASE_TRANS_DATA = "llm_base_trans_data"
    V1_2_BASE_COMB_DATA = "llm_base_comb_data"
    V1_3_SMALL_TRANS_DATA = "llm_small_trans_data"
    V1_4_SMALL_COMB_DATA = "llm_small_comb_data"

    V1_5_BASE_LM_DATA = "llm_base_lm_data"
    V1_6_SMALL_LM_DATA = "llm_small_lm_data"

    V2_1_BC_VOCAB = "llm_base_comb_vocab"
    V2_2_BC_VOCAB = "llm_small_comb_vocab"
    V2_3_BLLM_VOCAB = "llm_base_llm_vocab"
    V2_4_BLLM_VOCAB = "llm_small_llm_vocab"

    # Tests
    V1_BASELINE_TEST = "llm_baseline_test"


_EXPERIMENT_BUILDERS = {
    LLMExperimentVersion.V1_1_BASE_TRANS_DATA: exp_1_1,
    LLMExperimentVersion.V1_2_BASE_COMB_DATA: exp_1_2,
    LLMExperimentVersion.V1_3_SMALL_TRANS_DATA: exp_1_3,
    LLMExperimentVersion.V1_4_SMALL_COMB_DATA: exp_1_4,

    LLMExperimentVersion.V1_5_BASE_LM_DATA: exp_1_5,
    LLMExperimentVersion.V1_6_SMALL_LM_DATA: exp_1_6,

    LLMExperimentVersion.V2_1_BC_VOCAB: base_comb_sllm_vocab,
    LLMExperimentVersion.V2_2_BC_VOCAB: small_comb_sllm_vocab,
    LLMExperimentVersion.V2_3_BLLM_VOCAB: base_llm_sllm_vocab,
    LLMExperimentVersion.V2_4_BLLM_VOCAB: small_llm_sllm_vocab,

    # Tests
    LLMExperimentVersion.V1_BASELINE_TEST: exp_baseline_test,
}


def get_experiment_config(name: LLMExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
