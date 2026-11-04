"""Granite / Parakeet + Qwen: dev-only scale, size and speed experiments."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.qwen_tuning import run
    from .results import py as reports

    run()
    reports()


def main():
    py()
