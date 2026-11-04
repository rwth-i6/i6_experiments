"""Own BPE CTC/RNA models + dynamic Transformer/Qwen word fusion, dev only."""


def py() -> None:
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.wordwise_lm import run

    run()
    from .results import py as register_reports

    register_reports()


def main() -> None:
    py()
