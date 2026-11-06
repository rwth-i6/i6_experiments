from recipe.experiments.librispeech.groups import run_digital


def py():
    return run_digital("full_ctx_transducer", "qat_full_model", 8)
