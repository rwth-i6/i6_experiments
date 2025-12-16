from enum import Enum

from .experiment_config import exp_baseline, exp_v2, ExperimentConfig, exp_v3, exp_v4


class ExperimentVersion(Enum):
    V1_BASELINE = "baseline"
    V2_DROPOUT = "SLLM_dropout"
    V3_TUNED = "SLLM_tuned_dropout"
    V4_SMALL_DECODER = "SLLM_small_decoder"
    # Expand here


_EXPERIMENT_BUILDERS = {
    ExperimentVersion.V1_BASELINE: exp_baseline,
    ExperimentVersion.V2_DROPOUT: exp_v2,
    ExperimentVersion.V3_TUNED: exp_v3,
    ExperimentVersion.V4_SMALL_DECODER: exp_v4,
    # Expand here
}


def get_experiment_config(name: ExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
