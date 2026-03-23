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


def e3v5(
        last: bool = True,
        best: bool = False,
        best4: bool = False,
):
    ex3.sllm_ep(
        [ExperimentVersion.V5_LINEAR_ADAPTER],
        run_best=best,
        run_best_4=best4,
        only_specific_epochs=not last,
        run_only_last=last,
        specific_recognition_epochs=set({})
    )  # , itc_training=True)


def e3v6():
    ex3.sllm_ep([ExperimentVersion.V6_SMALL_DECODER_150kBS])  # , itc_training=True)


# ++++

# def e3v7():
#    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT])#, itc_training=True)


def e3v7_ctc(
        last: bool = True,
        best: bool = False,
        best4: bool = False,
):
    ex3.sllm_ep(
        [ExperimentVersion.V7_TUNED_DROPOUT_CTC_GD],
        run_best=best,
        run_best_4=best4,
        only_specific_epochs=not last,
        run_only_last=last,
        specific_recognition_epochs=set({})
    )  # , itc_training=True)


def e3v7_beam():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_BEAM])  # , itc_training=True)

def e3v7_beam_ln():
    ex3.sllm_ep([ExperimentVersion.V7_TUNED_DROPOUT_BEAM_LN])  # , itc_training=True)


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
        # [ExperimentVersion.V7_TUNED_DROPOUT_150, ExperimentVersion.V7_TUNED_DROPOUT_200]
        [ExperimentVersion.V7_TUNED_DROPOUT_200],
        run_test=False,
        #run_best=False,
        #run_best_4=False,
        #run_only_dev_other=True,
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
    ex3.sllm_ep(
        [ExperimentVersion.V7_TUNED_DROPOUT_300],
        run_test=False,
        #run_best=False,
        #run_best_4=False,
        #run_only_dev_other=True,
    )  # , itc_training=True)


def e3_f1_baseline(
        last: bool = True,
        best: bool = False,
        best4: bool = False,
):
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV2_4GPU_10K],
        run_best=best,
        run_best_4=best4,
        only_specific_epochs=not last,
        run_only_last=last,
        specific_recognition_epochs=set({})
    )


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
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_C],
        run_test=False,  # TODO: for now
        #run_best=False,  # TODO: for now
        #run_best_4=False,  # TODO: for now
        #run_only_dev_other=True,  # TODO: for now
    )  # , itc_training=True)


def e3_pre7_f2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_D_B_C_F2])  # , itc_training=True)


def e3_pre8_f1():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_C_F1])  # , itc_training=True)


def e3_lora1():
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV3_DS_PRE_ED_B_C_LORA],
        # itc_training=True,
        run_only_last=False,
    )


def e3_lora2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV3_PRE_ED_S_C_LORA])  # , itc_training=True)


def e3_lora3():
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_BV3_PRE_ED_S_C_LORA_small]
    )  # , itc_training=True)


# ++++


def e3_pre9():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_LM])  # , itc_training=True)


def e3_pre10():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_LM])  # , itc_training=True)


# +++


def e3v13_2():
    ex3.sllm_ep([ExperimentVersion.V13_SLLM_linear_adapter_200])  # , itc_training=True)


def e3v14():
    ex3.sllm_ep(
        [ExperimentVersion.V14_SLLM_3CTC],
        run_test=False,
        # run_best=False,
        # run_best_4=False,
        # run_only_dev_other=True,
    )  # , itc_training=True)


def e3_ft1():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_LR4])  # , itc_training=True)


def e3_ft2():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_LR5])  # , itc_training=True)


# +++


def e3v7_s2():
    ex3.sllm_ep([ExperimentVersion.V7_S2])  # , itc_training=True)


def e3v7_s2v2():
    ex3.sllm_ep([ExperimentVersion.V7_S2V2])  # , itc_training=True)


def e3v5_s2():
    ex3.sllm_ep([ExperimentVersion.V5_LINEAR_ADAPTER_S2])  # , itc_training=True)


def e3v5_s2v2():
    ex3.sllm_ep([ExperimentVersion.V5_LINEAR_ADAPTER_S2V2])  # , itc_training=True)


def e3v10_3_s2():
    ex3.sllm_ep([ExperimentVersion.V10_SMALL_DECODER_4GPUS_V3_S2])


# +++


def e3_ft3_i6():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA5_I6])


def e3_ft4_i6():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA4_I6])


def e3_ft3():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA5])  # , itc_training=True)


def e3_ft4():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_CA4])  # , itc_training=True)


# +++


def e3_pre11():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_B_FE])  # , itc_training=True)


def e3_pre12():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_FE])


def e3_itc_config_test():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_C_1], itc_training=True)


# +++


def e3v14_pre1():
    ex3.sllm_ep([ExperimentVersion.V14_SLLM_3CTC_B_PRE_ED])  # , itc_training=True)


def e3v14_pre2():
    ex3.sllm_ep([ExperimentVersion.V14_SLLM_3CTC_S_PRE_ED])


def e3_pre5_i6():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_I6])


def e3v14_pre1_f20():
    ex3.sllm_ep([ExperimentVersion.SLLM_3CTC_BV2_DS_PRE_ED_B_C_F20], itc_training=True)


def e3v14_pre2_f10():
    ex3.sllm_ep([ExperimentVersion.SLLM_3CTC_BV2_PRE_ED_S_C_F10])


# NEW LLM PRETRAININGS!


def e3v14_pre1_v2():
    ex3.sllm_ep([ExperimentVersion.V14_SLLM_3CTC_B_PRE_ED_V2], itc_training=True)


def e3v14_pre2_v2(
        last: bool = False,
        best: bool = True,
        best4: bool = False,
):
    ex3.sllm_ep([ExperimentVersion.V14_SLLM_3CTC_S_PRE_ED_V2],
                run_best=best,
                run_best_4=best4,
                only_specific_epochs=not last,
                run_only_last=last,
                specific_recognition_epochs=set({})
                )


def e3_pre5_i6_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_I6_V2])


def e3v14_pre1_f20_v2(
        last: bool = False,
        best: bool = False,
        best4: bool = True,
):
    ex3.sllm_ep(
        [ExperimentVersion.SLLM_3CTC_BV2_DS_PRE_ED_B_C_F20_V2],
        run_best=best,
        run_best_4=best4,
        only_specific_epochs=not last,
        run_only_last=last,
        specific_recognition_epochs=set({})
        #, itc_training=True
    )

def e3v14_pre1_f40_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_3CTC_BV2_DS_PRE_ED_B_C_F40_V2], itc_training=True)


def e3v14_pre2_f10_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_3CTC_BV2_PRE_ED_S_C_F10_V2])


def e3_pre11_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_B_FE_V2])#, itc_training=True)


def e3_pre12_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_FE_V2])


def e3_pre15():
    ex3.sllm_ep([ExperimentVersion.V15_SLLM_3CTC_SMALL])

def e3_pre15_hpc():
    ex3.sllm_ep([ExperimentVersion.V15_SLLM_3CTC_SMALL_HPC], itc_training=True)

def e3_pre5_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_C_V2])  # , itc_training=True)


def e3_pre10_v2(
        last: bool = True,
        best: bool = False,
        best4: bool = False,
):
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_ED_S_LM_V2],
                run_best=best,
                run_best_4=best4,
                only_specific_epochs=not last,

                specific_recognition_epochs=set({})
                )  # , itc_training=True)


def e3_ft1_v2():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_LR4_V2])  # , itc_training=True)


def e3_ft2_v2():
    ex3.sllm_ep([ExperimentVersion.V15_SMALL_SLLM_LR5_V2])  # , itc_training=True)


def e3_pre8_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_C_V2], itc_training=True)


def e3_pre9_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_DS_PRE_ED_B_LM_V2], itc_training=True)


def e3_pre3_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_S_C_V2], itc_training=True)

def e3_pre3_v2_i6():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_S_C_V2_I6])


def e3_pre4_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV2_PRE_D_B_C_V2])#, itc_training=True)


# LORA V2


def new_lora():
    e3_lora1_v2()
    e3_lora2_v2()
    e3_lora3_v2()

def e3_lora1_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV3_DS_PRE_ED_B_C_LORA_V2])#, itc_training=True)


def e3_lora2_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV3_PRE_ED_S_C_LORA_V2])#, itc_training=True)


def e3_lora3_v2():
    ex3.sllm_ep([ExperimentVersion.SLLM_BV3_PRE_ED_S_C_LORA_small_V2])#, itc_training=True)


# BEST POSSIBLE MODELS

def best_pretrained():
    e3_best_large()
    e3_best_small()

def e3_best_large():
    ex3.sllm_ep([ExperimentVersion.SLLM_3CTC_B_PRE_ED_C_F20], itc_training=True) # Already run!! 

def e3_best_small():
    ex3.sllm_ep([ExperimentVersion.SLLM_3CTC_S_PRE_ED_LM], itc_training=True)


"""
SPECIAL
"""

def final_v4_decoding():
    e3v7_ctc()
    e3v5()
    e3_f1_baseline()

    # Pretrainings
    e3v14_pre1_f20_v2()
    e3_pre10_v2()
    e3v14_pre2_v2()







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


def v4_ctc_sllm_decoding():
    e3v7_ctc()
    e3v8()

    e3v6()
    e3v9()
    e3v10_2()
    e3v10()
    e3v10_3()
    e3v5()

    # e3v11() # LAST CHECKPOINT NOT FOUND!
    e3v12()
    e3v13()
    e3v8_2()
    e3v7_2()
    e3v7_3()

    #e3_pre3() # BROKEN
    #e3_pre4() # BROKEN
    #e3_pre5() # BROKEN
    #e3_pre6() # BROKEN

    e3_f1_baseline()
    #e3_f1_baseline_pre() # BROKEN
    #e3_f1() # BROKEN
    #e3_f1_ds() # BROKEN
    #e3_f2() # BROKEN
    #e3_f3() # BROKEN
    #e3_f4() # BROKEN
    #e3_f5() # BROKEN
    #e3_f5_2oclr_2() # BROKEN

    #e3_lora1() # BROKEN
    #e3_lora2() # BROKEN
    #e3_lora3() # BROKEN

    e3v13_2()
    e3v14()

    e3v7_beam()
    e3v7_beam_ln()

    # e3_ft1() # BROKEN
    # e3_ft2() # BROKEN
    # e3_ft3_i6() # BROKEN
    # e3_ft4_i6() # BROKEN
    # e3_ft3() # BROKEN
    # e3_ft4() # BROKEN

    e3v7_s2()
    e3v7_s2v2()
    e3v5_s2()
    e3v5_s2v2()
    e3v10_3_s2()

    # e3_pre11() # BROKEN
    # e3_pre12() # BROKEN

    #e3v14_pre1() # BROKEN
    #e3v14_pre2() # BROKEN
    e3_pre5_i6()
    e3v14_pre2_f10()

    e3v14_pre1_v2()
    e3v14_pre2_v2()
    e3_pre5_i6_v2()
    e3v14_pre1_f20_v2()
    e3v14_pre2_f10_v2()
    e3_pre11_v2()
    e3_pre12_v2()
    # e3_pre15() TODO: running i6
    e3_pre5_v2()
    e3_pre10_v2()
    e3_ft1_v2()
    e3_ft2_v2()
    e3_pre8_v2()
    e3_pre9_v2()
    e3_pre3_v2()
    # e3_pre3_v2_i6() # TODO: running i6
    e3_pre4_v2()

    e3_lora2_v2()
    e3_lora3_v2()


def new_v4_search():
    e3v7_ctc()
    e3v5()
    e3_pre5()
    e3_pre8()


def new_v4_search_v2():
    e3v7_ctc()
    e3_pre8()
    e3v14()


def new_v4_len_norm():
    e3v5()
    e3v7_ctc()
    e3v14()


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


def param_test():
    ex3.sllm_ep([
        ExperimentVersion.PARAM_TEST_2CTC,
        ExperimentVersion.PARAM_TEST_3CTC,
    ], debug=True)



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

if __name__ == "__main__":  # For debugging purposes
    e3v5()
