"""
Search RTF vs. WER curves for fixed and dynamic pruning with the BPE transducer and the
word-level 4-gram LM on dev.all. This figure was cut from the final paper for space, but the
experiments are kept here since they are referenced by the Loquacious discussion in Sec. 5.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.loquacious import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []

    for score_threshold in [None, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for max_beam_size in [2, 4, 8, 16, 32, 64, 128]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

            variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_gpu"
            variant.search_mode_params.gpu_mem_rqmt = 24
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.medium",
            variants=variants,
            corpora=["dev.all"],
        )
    )

    return recog_results
