"""CPU-only Step 2; no inference, test-set tuning or best-on-test selection."""


def py():
    import os
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.open_asr.error_complementarity import run
    return run(protocol_file=os.environ.get("ASR_CONTROLLED_PROTOCOL"),
               selected_evidence=os.environ.get("ASR_COMPLEMENTARITY_EVIDENCE"))


def main():
    py()
