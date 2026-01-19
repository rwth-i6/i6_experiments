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


"""
Experiments entry points
"""


def e3():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE])


def e3_itc():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE])  #, itc_training=True)


# ++++


def e3_d():
    ex3.sllm_ep([ExperimentVersion.V2_DROPOUT])  #, itc_training=False)


def e3_td():
    raise RuntimeError("THIS EXPERIMENT SHOULD PROBABLY NOT BE RUN!")
    ex3.sllm_ep([ExperimentVersion.V3_TUNED], itc_training=True)


# ++++


def e3v4():
    ex3.sllm_ep([ExperimentVersion.V4_SMALL_DECODER])  #, itc_training=True)


def e3v5():
    ex3.sllm_ep([ExperimentVersion.V5_LINEAR_ADAPTER])  #, itc_training=True)


def e3v6():
    ex3.sllm_ep([ExperimentVersion.V6_SMALL_DECODER_150kBS])  #, itc_training=True)


# ++++

# def e3v7():
#    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT])#, itc_training=True)


def e3v7_ctc():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_CTC_GD])  # , itc_training=True)


def e3v7_beam():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_BEAM])  # , itc_training=True)


def e3v8():
    ex3.sllm_ep([ExperimentVersion.V8_1_TD_LRV2, ExperimentVersion.V8_2_TD_LRV3])  # , itc_training=True)


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
    ex3.sllm_ep([ExperimentVersion.V8_3_TD_LRV4, ExperimentVersion.V8_4_TD_LRV5])  # , itc_training=True)


def e3v7_2():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_150, ExperimentVersion.V7_TUNED_DROPOUT_200], itc_training=True)


"""
SPECIAL
"""


def v2_decoding():
    #e3()
    #e3_itc()

    # e3_d()
    # e3_td()

    #e3v4()
    e3v5()
    e3v6()

    e3v7_ctc()
    e3v8()

    e3v9()
    e3v10()
    e3v10_2()
    # e3v10_3() # TODO: running - do it later

    e3v11()
    e3v12()
    # e3v3_2()
    e3v13()
    # e3v2_s2()
    e3v8_2()
    # e3v7_2() # TODO: running - do it later


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
]
