"""Load only Parakeet RNN-T/Qwen development tuning, never the full suite."""


def py() -> None:
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.rnnt_qwen import run

    # Already prefixes aliases with open-asr/qwen-parakeet-rnnt/.
    run(tuning_only=True, focused_tuning=True)
    from .results import py as register_reports

    register_reports()


def main() -> None:
    py()
