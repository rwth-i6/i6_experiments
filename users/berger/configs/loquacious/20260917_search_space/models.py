"""
The three Loquacious models the SLT 2026 search-space comparison reports on, all trained on
the public medium (2500h) portion.

The word-level 4-gram LM is not trained here: it is pulled in on demand by the recognition
variants via `seq2seq_rasr_2025.data.loquacious.lm`.
"""

__all__ = ["run"]

from typing import Dict

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.loquacious import training
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run() -> Dict[str, TrainedModel]:
    return {
        "ctc_bpe": training.medium.ctc_bpe.run(descriptor="ctc_bpe"),
        "ffnn_transducer_bpe": training.medium.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_phoneme": training.medium.ffnn_transducer_phoneme.run(descriptor="ffnn_transducer_phoneme"),
    }
