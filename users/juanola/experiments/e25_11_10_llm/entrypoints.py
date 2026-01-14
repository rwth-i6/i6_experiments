from i6_experiments.users.juanola.experiments.e25_11_10_llm import experiments as ex4
from i6_experiments.users.juanola.experiments.e25_11_10_llm.configurations.experiment_version import \
    LLMExperimentVersion


def main():
    """
    main (called by sisyphus if nothing else is specified)

    Should not be used directly but contain all (non-test) subgraphs for easier cleanup.
    """
    e4()


"""
Experiments entry points
"""


def e4():
    ex4.llm_ep([LLMExperimentVersion.V1_BASELINE])


"""
DEBUGS
"""


def e4d():
    ex4.llm_ep([LLMExperimentVersion.V1_BASELINE_TEST], debug=True)


"""
Tests entry points
"""

__all__ = ["main", "e4"]
