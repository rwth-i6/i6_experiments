from i6_experiments.example_setups.librispeech.seq2seq_rasr_2024.config_ctc_bpe import py as py_ctc_bpe
from i6_experiments.example_setups.librispeech.seq2seq_rasr_2024.config_transducer_bpe import py as py_transducer_bpe


def main() -> None:
    py_ctc_bpe()
    py_transducer_bpe()
