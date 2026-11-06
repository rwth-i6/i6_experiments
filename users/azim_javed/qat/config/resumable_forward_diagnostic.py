"""Two independent, identically configured digital QAT lexfree decodings."""

from recipe.experiments.librispeech.diagnostics.resumable_forward import run


def py():
    return run()
