"""CPU-only, frozen development selection and paired benchmark comparisons."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.controlled_comparison import run
    run()


def main():
    py()
