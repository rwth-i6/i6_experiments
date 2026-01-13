from i6_experiments.users.juanola.experiments.e25_11_13_ctc import experiments as ex5
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.experiment_version import \
    CTCExperimentVersion


def main():
    """
    main (called by sisyphus if nothing else is specified)

    Should not be used directly but contain all (non-test) subgraphs for easier cleanup.
    """
    e5()

"""
Experiments entry points
"""

def e5():
    ex5.ctc_ep([CTCExperimentVersion.V1_BASELINE])


"""
DEBUGS
"""

def e5d():
    ex5.ctc_ep([CTCExperimentVersion.V1_BASELINE], debug=True)


"""
Tests entry points
"""



__all__ = ["main", "e5"]


