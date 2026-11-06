from recipe.experiments.librispeech.groups import run_digital


def py():
    return run_digital("ffnn_transducer", "qat_full_model", 4)
