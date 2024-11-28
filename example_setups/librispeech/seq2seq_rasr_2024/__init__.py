from i6_experiments.example_setups.librispeech.seq2seq_rasr_2024.bpe_ctc.pipeline import run as run_bpe_ctc
from i6_experiments.example_setups.librispeech.seq2seq_rasr_2024.bpe_transducer.pipeline import (
    run as run_bpe_transducer,
)


def main() -> None:
    run_bpe_ctc()
    run_bpe_transducer()
