"""Step 4 only: fixed duration/source/frequency slices, CPU-only."""


def py():
    import os
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.research_diagnostics import run

    return run(
        which="stratification",
        protocol_file=os.environ.get("ASR_CONTROLLED_PROTOCOL"),
        selected_evidence=os.environ.get("ASR_COMPLEMENTARITY_EVIDENCE"),
        training_corpus=os.environ.get("ASR_RARITY_TRAINING_CORPUS"),
    )


def main():
    py()
