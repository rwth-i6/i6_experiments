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

    # DEBUGS
    e3d()
    e3d_itc()

    # TESTS
    t1()


"""
Experiments entry points
"""


def e3():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE])


def e3_itc():
    ex3.sllm_ep([ExperimentVersion.V1_BASELINE], itc_training=True)


def e3_d():
    ex3.sllm_ep([ExperimentVersion.V2_DROPOUT], itc_training=False)  # TODO: run this?


def e3_td():
    ex3.sllm_ep([ExperimentVersion.V3_TUNED], itc_training=True)


def e3v4():
    ex3.sllm_ep([ExperimentVersion.V4_SMALL_DECODER], itc_training=True)


def e3v5():
    ex3.sllm_ep([ExperimentVersion.V5_LINEAR_ADAPTER], itc_training=True)


def e3v6():
    ex3.sllm_ep([ExperimentVersion.V6_SMALL_DECODER_150kBS], itc_training=True)


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


__all__ = ["main", "e3", "e3_itc", "e3_d", "e3_td", "e3d", "e3d_itc", "e3v4", "e3v5", "e3v6"]
