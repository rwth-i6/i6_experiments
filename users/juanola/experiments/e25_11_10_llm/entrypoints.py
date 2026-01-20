from i6_experiments.users.juanola.experiments.e25_11_10_llm import experiments as ex4
from i6_experiments.users.juanola.experiments.e25_11_10_llm.configurations.experiment_version import \
    LLMExperimentVersion
from i6_experiments.users.juanola.experiments.e25_11_10_llm import tests as t4


def main():
    """
    main (called by sisyphus if nothing else is specified)

    Should not be used directly but contain all (non-test) subgraphs for easier cleanup.
    """
    e4_bt()
    e4_bc()
    e4_st()
    e4_sc()


"""
Experiments entry points
"""


def e4_bt():
    ex4.llm_ep([LLMExperimentVersion.V1_1_BASE_TRANS_DATA], only_specific_epochs=True)  # Avoid any recog for now (not implemented yet)

def e4_bc():
    ex4.llm_ep([LLMExperimentVersion.V1_2_BASE_COMB_DATA], only_specific_epochs=True)  # Avoid any recog for now (not implemented yet)

def e4_st():
    ex4.llm_ep([LLMExperimentVersion.V1_3_SMALL_TRANS_DATA], only_specific_epochs=True)  # Avoid any recog for now (not implemented yet)

def e4_sc():
    ex4.llm_ep([LLMExperimentVersion.V1_4_SMALL_COMB_DATA], only_specific_epochs=True)  # Avoid any recog for now (not implemented yet)


"""
DEBUGS
"""


def e4d():
    ex4.llm_ep([LLMExperimentVersion.V1_BASELINE_TEST], debug=True)


"""
Tests entry points
"""


def shuffle_job():
    t4.shuffle_file_test()


def split_job():
    t4.two_way_split_file_test()


__all__ = ["main", "e4_bt", "e4_bc", "e4_st", "e4_sc"]
