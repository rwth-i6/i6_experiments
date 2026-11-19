"""Parakeet TDT v3 native and RASR fusion development benchmarks."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.parakeet_tdt_experiments import run

    return run(gs.OUTPUT_DIR)
