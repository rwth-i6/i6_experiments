"""Dev-only Granite prior / Parakeet ILM grids and stronger Speech-ASR controls."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.pretrained_extensions import run
    from .results import py as reports
    run()
    reports()


def main():
    py()
