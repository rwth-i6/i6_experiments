from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2 import (
    experiments as ex3,
)
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2 import tests as t3
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.experiment_version import (
    ExperimentVersion,
)


def main():
    """
    main (called by sisyphus if nothing else is specified)

    Should not be used directly but contain all (non-test) subgraphs for easier cleanup.
    """
    e3()
    e3_itc()

    e3_d()
    e3_td()

    e3v4()
    e3v5()
    e3v6()

    # e3v7()
    e3v8()

    e3v9()
    e3v10()
    e3v10_2()
    e3v10_3()

    e3v11()
    e3v12()
    e3v3_2()
    e3v13()
    e3v2_s2()
    e3v8_2()
    e3v7_2()

    # e3_pre1()
    # e3_pre2()
    e3_pre3()
    e3_pre4()
    e3_pre5()
    e3_pre6()

    e3v7_3()

    e3_f1_baseline_pre()
    e3_f1_ds()

    e3_f1()
    e3_f2()
    e3_f3()
    e3_f4()
    e3_f5()

    e3_f5_2oclr_1()
    e3_f5_2oclr_2()

    e3_pre7()
    e3_pre8()
    e3_pre7_f2()
    e3_pre8_f1()

    e3_lora1()
    e3_lora2()
    e3_lora3()


"""
Experiments entry points
"""


def e3():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE])


def e3_itc():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE])  # , itc_training=True)


# ++++


def e3_d():
    ex3.sllm_ep([ExperimentVersion.V2_DROPOUT])  # , itc_training=False)


def e3_td():
    raise RuntimeError("THIS EXPERIMENT SHOULD PROBABLY NOT BE RUN!")
    ex3.sllm_ep([ExperimentVersion.V3_TUNED], itc_training=True)


# ++++


def e3v4():
    ex3.sllm_ep([ExperimentVersion.V4_SMALL_DECODER])  # , itc_training=True)


def e3v5():
    ex3.sllm_ep([ExperimentVersion.V5_LINEAR_ADAPTER])  # , itc_training=True)


def e3v6():
    ex3.sllm_ep([ExperimentVersion.V6_SMALL_DECODER_150kBS])  # , itc_training=True)


# ++++

# def e3v7():
#    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT])#, itc_training=True)


def e3v7_ctc():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_CTC_GD])  # , itc_training=True)


def e3v7_beam():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_BEAM])  # , itc_training=True)


def e3v8():
    ex3.sllm_ep(
        [ExperimentVersion.V8_1_TD_LRV2, ExperimentVersion.V8_2_TD_LRV3]
    )  # , itc_training=True)


# ++++


def e3v9():
    ex3.sllm_ep([ExperimentVersion.V9_SMALL_DECODER_250kBS])  # , itc_training=True,)


def e3v10():
    ex3.sllm_ep([ExperimentVersion.V10_SMALL_DECODER_4GPUS])


def e3v10_2():
    ex3.sllm_ep([ExperimentVersion.V10_SMALL_DECODER_4GPUS_V2])


def e3v10_3():
    ex3.sllm_ep([ExperimentVersion.V10_SMALL_DECODER_4GPUS_V3])


# ++++


def e3v11():
    ex3.sllm_ep([ExperimentVersion.V11_SLLM_D_80k])  # , itc_training=True)


def e3v12():
    ex3.sllm_ep([ExperimentVersion.V12_SLLM_t_80k])  # itc_training=True)


def e3v3_2():
    ex3.sllm_ep([ExperimentVersion.V3_SLLM_td_15k])  # , itc_training=True)


def e3v13():
    ex3.sllm_ep([ExperimentVersion.V13_SLLM_linear_adapter])  # , itc_training=True)


def e3v2_s2():
    ex3.sllm_ep([ExperimentVersion.V2_DROPOUT_s2])


def e3v8_2():
    ex3.sllm_ep(
        [ExperimentVersion.V8_3_TD_LRV4, ExperimentVersion.V8_4_TD_LRV5]
    )  # , itc_training=True)


def e3v7_2():
    ex3.sllm_ep(
        [ExperimentVersion.V7_TUNED_DROPOUT_150, ExperimentVersion.V7_TUNED_DROPOUT_200]
        #[ExperimentVersion.V7_TUNED_DROPOUT_200]
    )  # , itc_training=True)


# ++++


def e3_pre1():
    """
    Not used
    """
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_B_T], itc_training=True)


def e3_pre2():
    """
    Not used
    """
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_B_T], itc_training=True)


def e3_pre3():
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV2_PRE_D_S_C], run_only_last=False
    )  # , itc_training=True)


def e3_pre4():
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV2_PRE_D_B_C], run_only_last=False
    )  # , itc_training=True)


def e3_pre5():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C])  # , itc_training=True)


def e3_pre6():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_B_C])  # , itc_training=True)


# ++++


def e3v7_3():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_300])#, itc_training=True)


def e3_f1_baseline():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_4GPU_10K], run_only_last=False)


def e3_f1_baseline_pre():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_4GPU_10K_PRE_D_S_C], run_only_last=False)


def e3_f1():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_S_C_F1], run_only_last=False)


def e3_f2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_S_C_F2])


def e3_f3():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_S_C_F5])


def e3_f4():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_F1])


def e3_f5():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_F2])


def e3_f5_2oclr_1():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_F2_2OCLR_1])


def e3_f5_2oclr_2():
    """Depends on e3_f5_2oclr_1"""
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_F2_2OCLR_2])


def e3_f1_ds():
    """
    to test that old setup with 13k bs works with DS
    :return:
    """
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_D_S_C_F1], run_only_last=False)


def e3_pre7():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_D_B_C])  # , itc_training=True)


def e3_pre8():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_C])  # , itc_training=True)


def e3_pre7_f2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_D_B_C_F2])  # , itc_training=True)


def e3_pre8_f1():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_C_F1])  # , itc_training=True)


def e3_lora1():
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV3_DS_PRE_ED_B_C_LORA],
        #itc_training=True,
        run_only_last=False,
    )


def e3_lora2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV3_PRE_ED_S_C_LORA])#, itc_training=True)


def e3_lora3():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV3_PRE_ED_S_C_LORA_small])#, itc_training=True)


# ++++


def e3_pre9():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_LM])#, itc_training=True)


def e3_pre10():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_LM])#, itc_training=True)


# +++

def e3v13_2():
    ex3.sllm_ep([ExperimentVersion.V13_SLLM_linear_adapter_200], itc_training=True)

def e3v14():
    ex3.sllm_ep([ExperimentVersion.V14_SLLM_3CTC], itc_training=True)

def e3_ft1():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_LR4])#, itc_training=True)

def e3_ft2():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_LR5])#, itc_training=True)

# +++

def e3v7_s2():
    ex3.sllm_ep([ExperimentVersion.V7_S2], itc_training=True)


def e3v5_s2():
    ex3.sllm_ep([ExperimentVersion.V5_LINEAR_ADAPTER_S2], itc_training=True)


def e3v10_3_s2():
    ex3.sllm_ep([ExperimentVersion.V10_SMALL_DECODER_4GPUS_V3_S2])

# +++


def e3_ft3_i6():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA5_I6])

def e3_ft4_i6():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA4_I6])

def e3_ft3():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA5], itc_training=True)

def e3_ft4():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA4], itc_training=True)




"""
SPECIAL
"""


def v2_decoding():
    # e3() # TODO: old commit
    # e3_d() # Diverged
    # e3_td() # BUG

    # e3v4() # Diverged
    e3v5()
    e3v6()

    e3v7_ctc()
    e3v8()

    e3v9()
    e3v10()
    e3v10_2()
    e3v10_3()

    e3v11()
    e3v12()
    # e3v3_2() # Diverged
    e3v13()
    # e3v2_s2() # Diverged
    e3v8_2()
    e3v7_2()

def new_v4_search():
    e3v7_ctc()
    e3v5()
    e3_pre5()
    e3_pre8()



"""
DEBUGS
"""


def e3d():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE], debug=True)


def e3d_itc():
    ex3.sllm_ep(
        [ExperimentVersion.V1_BASELINE],
        debug=True,
        itc_training=True,
    )


"""
Tests entry points
"""


def t1():
    t3.hf_config_download_test()


def report_t1():
    t3.report_job_test_with_results()


def report_t2():
    t3.report_job_test_with_results_and_template()


def report_t3():
    t3.report_job_test_register_report()


def real_size_test():
    ex3.sllm_ep([ExperimentVersion.V10_SMALL_DECODER_4GPUS_V2], debug=True)


def n2_test():
    ex3.sllm_ep([ExperimentVersion.N2_TEST], specific_recognition_epochs={2})


def n2_test2():
    ex3.sllm_ep([ExperimentVersion.N2_TEST_SV2], specific_recognition_epochs={2})


__all__ = [
    "main",
    "e3",
    "e3_itc",
    "e3_d",
    "e3_td",
    "e3d",
    "e3d_itc",
    "e3v4",
    "e3v5",
    "e3v6",
    "e3v8",
    "e3v9",
    "e3v10",
    "e3v10_2",
    "e3v10_3",
    "e3v11",
    "e3v12",
    "e3v3_2",
    "e3v13",
    "e3v2_s2",
    "e3v8_2",
    "e3v7_2",
    # "e3_pre1",
    # "e3_pre2",
    "e3_pre3",
    "e3_pre4",
    "e3_pre5",
    "e3_pre6",
    "e3v7_3",
    "e3_f1_baseline_pre",
    "e3_f1_ds",
    "e3_f1",
    "e3_f2",
    "e3_f3",
    "e3_f4",
    "e3_f5",
    "e3_f5_2oclr_1",
    "e3_f5_2oclr_2",
    "e3_pre7",
    "e3_pre8",
    "e3_pre7_f2",
    "e3_pre8_f1",
    "e3_lora1",
    "e3_lora2",
    "e3_lora3",
]
