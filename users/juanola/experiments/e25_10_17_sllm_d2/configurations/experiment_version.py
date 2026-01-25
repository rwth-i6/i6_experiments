from enum import Enum

from .experiment_config import exp_baseline, exp_v2, ExperimentConfig, exp_v3, exp_v4, exp_v5, exp_v6, exp_v8_1, \
    exp_v8_2, \
    exp_v9, exp_v10, exp_v10_2, exp_v11, exp_v12, exp_v3_2, exp_v13, exp_v2_s2, exp_v8_3, exp_v8_4, exp_v7_150, \
    exp_v7_200, t_v1, t_v1_2, exp_v7_with_ctc_gd, exp_v10_3, n2_test, n2_test_sv2, exp_v7_with_beam, bv2_pre_d_b_t, \
    bv2_pre_ed_b_t, bv2_pre_d_s_c, bv2_pre_d_b_c, bv2_pre_ed_b_c, bv2_pre_ed_s_c


class ExperimentVersion(Enum):
    """
    Avoid duplicate values -> will lead to merged params
    """
    V1_BASELINE = "baseline"
    V2_DROPOUT = "SLLM_dropout"
    V3_TUNED = "SLLM_tuned_dropout"

    V4_SMALL_DECODER = "SLLM_small_decoder"
    V5_LINEAR_ADAPTER = "SLLM_small_linear_adapter"
    V6_SMALL_DECODER_150kBS = "SLLM_small_decoder_150kBS"

    #V7_TUNED_DROPOUT = "SLLM_tuned_dropout_v2" # Use the CTC one
    V7_TUNED_DROPOUT_CTC_GD = "SLLM_tuned_dropout_v2"
    V7_TUNED_DROPOUT_BEAM = "SLLM_tuned_dropout_v2_forward2"
    V8_1_TD_LRV2 = "SLLM_td_lrv2"
    V8_2_TD_LRV3 = "SLLM_td_lrv3"

    V9_SMALL_DECODER_250kBS = "SLLM_small_decoder_250kBS"
    V10_SMALL_DECODER_4GPUS = "SLLM_small_decoder_4gpus_i6"
    V10_SMALL_DECODER_4GPUS_V2 = "SLLM_small_decoder_4gpus_i6_v2"
    V10_SMALL_DECODER_4GPUS_V3 = "SLLM_small_decoder_4gpus_i6_v3"

    V11_SLLM_D_80k = "SLLM_dropout_80k"
    V12_SLLM_t_80k = "SLLM_tuned_80k"
    V3_SLLM_td_15k = "SLLM_td_15k"
    V13_SLLM_linear_adapter = "SLLM_linear_adapter"
    V2_DROPOUT_s2 = "SLLM_dropout_s2"
    V8_3_TD_LRV4 = "SLLM_td_lrv4"
    V8_4_TD_LRV5 = "SLLM_td_lrv5"
    V7_TUNED_DROPOUT_150= "SLLM_td_150"
    V7_TUNED_DROPOUT_200= "SLLM_td_200"

    SLLM_BV2_PRE_D_B_T = "SLLM_pretrained_d_b_t"
    SLLM_BV2_PRE_ED_B_T = "SLLM_pretrained_ed_b_t"
    SLLM_BV2_PRE_D_S_C = "SLLM_pretrained_d_s_c"
    SLLM_BV2_PRE_D_B_C = "SLLM_pretrained_d_b_c"
    SLLM_BV2_PRE_ED_S_C = "SLLM_pretrained_ed_s_c"
    SLLM_BV2_PRE_ED_B_C = "SLLM_pretrained_ed_b_c"

    # Expand here

    # Tests
    T1_CTC_GREEDY_DECODING = "test1-ctc-greedy-decoding"
    T1_2_CTC_GREEDY_DECODING = "test1-ctc-greedy-decoding_10kBS"
    N2_TEST = "n2_test"
    N2_TEST_SV2 = "n2_test_sv2"


_EXPERIMENT_BUILDERS = {
    ExperimentVersion.V1_BASELINE: exp_baseline,
    ExperimentVersion.V2_DROPOUT: exp_v2,
    ExperimentVersion.V3_TUNED: exp_v3,

    ExperimentVersion.V4_SMALL_DECODER: exp_v4,
    ExperimentVersion.V5_LINEAR_ADAPTER: exp_v5,
    ExperimentVersion.V6_SMALL_DECODER_150kBS: exp_v6,

    #ExperimentVersion.V7_TUNED_DROPOUT: exp_v7,
    ExperimentVersion.V7_TUNED_DROPOUT_CTC_GD: exp_v7_with_ctc_gd,
    ExperimentVersion.V7_TUNED_DROPOUT_BEAM: exp_v7_with_beam,
    ExperimentVersion.V8_1_TD_LRV2: exp_v8_1,
    ExperimentVersion.V8_2_TD_LRV3: exp_v8_2,

    ExperimentVersion.V9_SMALL_DECODER_250kBS: exp_v9,
    ExperimentVersion.V10_SMALL_DECODER_4GPUS: exp_v10,
    ExperimentVersion.V10_SMALL_DECODER_4GPUS_V2: exp_v10_2,
    ExperimentVersion.V10_SMALL_DECODER_4GPUS_V3: exp_v10_3,

    ExperimentVersion.V11_SLLM_D_80k: exp_v11,
    ExperimentVersion.V12_SLLM_t_80k: exp_v12,
    ExperimentVersion.V3_SLLM_td_15k: exp_v3_2,
    ExperimentVersion.V13_SLLM_linear_adapter: exp_v13,
    ExperimentVersion.V2_DROPOUT_s2: exp_v2_s2,
    ExperimentVersion.V8_3_TD_LRV4: exp_v8_3,
    ExperimentVersion.V8_4_TD_LRV5: exp_v8_4,
    ExperimentVersion.V7_TUNED_DROPOUT_150: exp_v7_150,
    ExperimentVersion.V7_TUNED_DROPOUT_200: exp_v7_200,

    ExperimentVersion.SLLM_BV2_PRE_D_B_T: bv2_pre_d_b_t,
    ExperimentVersion.SLLM_BV2_PRE_ED_B_T: bv2_pre_ed_b_t,
    ExperimentVersion.SLLM_BV2_PRE_D_S_C: bv2_pre_d_s_c,
    ExperimentVersion.SLLM_BV2_PRE_D_B_C: bv2_pre_d_b_c,
    ExperimentVersion.SLLM_BV2_PRE_ED_S_C: bv2_pre_ed_s_c,
    ExperimentVersion.SLLM_BV2_PRE_ED_B_C: bv2_pre_ed_b_c,
    # Expand here

    # Tests
    ExperimentVersion.T1_CTC_GREEDY_DECODING: t_v1,
    ExperimentVersion.T1_2_CTC_GREEDY_DECODING: t_v1_2,
    ExperimentVersion.N2_TEST: n2_test,
    ExperimentVersion.N2_TEST_SV2: n2_test_sv2,
}


def get_experiment_config(name: ExperimentVersion) -> ExperimentConfig:
    return _EXPERIMENT_BUILDERS[name]()
