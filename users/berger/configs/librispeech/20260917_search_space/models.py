"""
The three LibriSpeech models the SLT 2026 search-space comparison reports on.

The external LMs (subword-level LSTM, subword- and word-level Transformer, and the official
4-gram) are not trained here: they are pulled in on demand by the recognition variants via
`seq2seq_rasr_2025.data.librispeech.lm`.
"""

__all__ = ["run"]

from typing import Dict

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.librispeech import training
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run() -> Dict[str, TrainedModel]:
    return {
        "ctc_bpe": training.ctc_bpe.run(descriptor="ctc_bpe"),
        "ctc_phoneme": training.ctc_phoneme.run(descriptor="ctc_phoneme"),
        "ffnn_transducer_bpe": training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
    }
