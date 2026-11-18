"""Four extra Speech-Qwen GPU scorers over existing dev n-best pools only."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.contraction_experiments import run

    return run(gs.OUTPUT_DIR)
