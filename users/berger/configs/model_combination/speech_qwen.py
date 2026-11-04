"""Dev-only Speech-Qwen balance/prior grids, n-best rescoring and thresholds."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.speech_qwen_round import run
    from .results import py as reports
    run()
    reports()


def main():
    py()
