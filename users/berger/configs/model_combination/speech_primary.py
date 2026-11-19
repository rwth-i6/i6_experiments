"""Speech-Qwen primary beam proposals with Granite CTC and Parakeet RNN-T rescoring."""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.speech_primary import run

    return run(gs.OUTPUT_DIR)
