"""Evaluate the completed frozen development selection; no new tuning grids."""


def py():
    import os
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.controlled_evaluation import run
    run(protocol_file=os.environ.get("ASR_CONTROLLED_PROTOCOL"))


def main():
    py()
