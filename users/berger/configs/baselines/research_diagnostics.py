"""CPU-only Steps 3 and 4: candidate ranking and difficulty stratification."""


def py():
    import os
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.research_diagnostics import run

    return run(
        protocol_file=os.environ.get("ASR_CONTROLLED_PROTOCOL"),
        selected_evidence=os.environ.get("ASR_COMPLEMENTARITY_EVIDENCE"),
        training_corpus=os.environ.get("ASR_RARITY_TRAINING_CORPUS"),
    )


def main():
    py()
