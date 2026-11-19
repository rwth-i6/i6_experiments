"""Parakeet TDT v3 native and RASR fusion development benchmarks.

If time permits, try the English-only Parakeet TDT 0.6B v2 checkpoint.
Compare standalone and n-best oracle WER on LibriSpeech dev-other and
Loquacious dev.short before deciding whether to switch the fusion runs.
"""


def py():
    from sisyphus import gs
    from i6_experiments.users.berger.model_pipelines.common.parakeet_tdt_experiments import run

    return run(gs.OUTPUT_DIR)
