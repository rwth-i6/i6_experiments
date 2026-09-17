"""
Table 5: WER, search errors and within-word hypothesis-count percentiles over the pruning
parameters, for BPE transducer lexicon-constrained search with the word-level 4-gram LM on
dev.all.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.loquacious import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []
    for score_threshold, max_beam_size in [(0.0, 1), (2.0, 16), (4.0, 32), (6.0, 32), (8.0, 32)]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        variant.search_algorithm_params.score_thresholds = [score_threshold]
        variant.compute_search_errors = True
        variants.append(variant)

        # variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        # variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_gpu_search_err"
        # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        # variant.search_algorithm_params.score_thresholds = [score_threshold]
        # variant.search_mode_params.gpu_mem_rqmt = 24
        # variant.compute_search_errors = True
        # variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.all"]
        )
    )

    return recog_results
