"""Parakeet + Speech-Qwen: Earnings22 pruning / speed diagnostic."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.earnings_qwen import run
    from .results import py as reports

    run()
    reports()


def main():
    py()
