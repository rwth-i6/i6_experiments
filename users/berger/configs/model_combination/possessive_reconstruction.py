"""Granite possessive restoration on both frozen development corpora."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.possessive_reconstruction import run

    return run(gs.OUTPUT_DIR)
