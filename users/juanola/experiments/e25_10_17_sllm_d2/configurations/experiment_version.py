from enum import Enum

from .experiment_config import exp_baseline, exp_v2, ExperimentConfig, exp_v3, exp_v4, exp_v5, exp_v6, exp_v7, exp_v8_1, \
    exp_v8_2, \
    exp_v9, exp_v10, exp_v10_2


class ExperimentVersion(Enum):
    V1_BASELINE = "baseline"
    V2_DROPOUT = "SLLM_dropout"
    V3_TUNED = "SLLM_tuned_dropout"

    V4_SMALL_DECODER = "SLLM_small_decoder"
    V5_LINEAR_ADAPTER = "SLLM_linear_adapter"
    V6_SMALL_DECODER_150kBS = "SLLM_small_decoder_150kBS"

    V7_TUNED_DROPOUT = "SLLM_tuned_dropout_v2"
    V8_1_TD_LRV2 = "SLLM_td_lrv2"
    V8_2_TD_LRV3 = "SLLM_td_lrv3"

    V9_SMALL_DECODER_250kBS = "SLLM_small_decoder_250kBS"
    V10_SMALL_DECODER_4GPUS = "SLLM_small_decoder_4gpus_i6"
    V10_SMALL_DECODER_4GPUS_V2 = "SLLM_small_decoder_4gpus_i6_v2"
    # Expand here


_EXPERIMENT_BUILDERS = {
    ExperimentVersion.V1_BASELINE: exp_baseline,
    ExperimentVersion.V2_DROPOUT: exp_v2,
    ExperimentVersion.V3_TUNED: exp_v3,

    ExperimentVersion.V4_SMALL_DECODER: exp_v4,
    ExperimentVersion.V5_LINEAR_ADAPTER: exp_v5,
    ExperimentVersion.V6_SMALL_DECODER_150kBS: exp_v6,

    ExperimentVersion.V7_TUNED_DROPOUT: exp_v7,
    ExperimentVersion.V8_1_TD_LRV2: exp_v8_1,
    ExperimentVersion.V8_2_TD_LRV3: exp_v8_2,

    ExperimentVersion.V9_SMALL_DECODER_250kBS: exp_v9,
    ExperimentVersion.V10_SMALL_DECODER_4GPUS: exp_v10,
    ExperimentVersion.V10_SMALL_DECODER_4GPUS_V2: exp_v10_2,
    # Expand here
}


def get_experiment_config(name: ExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
