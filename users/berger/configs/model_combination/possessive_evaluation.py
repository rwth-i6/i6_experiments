"""Frozen Granite possessive reconstruction on the public test suite."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.possessive_evaluation import run

    return run(gs.OUTPUT_DIR)
