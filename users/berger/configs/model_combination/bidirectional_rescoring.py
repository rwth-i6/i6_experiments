"""Run bidirectional n-best fusion with Speech Qwen 1.7B on the development sets."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.bidirectional_experiments import run

    return run(gs.OUTPUT_DIR)
