"""Step 3 only; no recognition and no training-frequency dependency."""


def py():
    import os
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.research_diagnostics import run

    return run(which="ranking", protocol_file=os.environ.get("ASR_CONTROLLED_PROTOCOL"))


def main():
    py()
