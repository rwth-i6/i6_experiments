from i6_experiments.users.juanola.experiments.e25_09_08_demo1_sllm.aed import aed as aed_base

def main():
    """
    main (called by sisyphus if nothing else is specified)

    Should not be used directly but contain all (non-test) subgraphs for easier cleanup.
    """
    aed()

"""
Experiments entry points
"""

def aed():
    aed_base.aed_baseline()


__all__ = ["main", "aed"]