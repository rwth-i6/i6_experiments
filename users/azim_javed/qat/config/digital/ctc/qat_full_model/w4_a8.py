from recipe.experiments.librispeech.groups import run_digital


def py():
    return run_digital("ctc", "qat_full_model", 4)
