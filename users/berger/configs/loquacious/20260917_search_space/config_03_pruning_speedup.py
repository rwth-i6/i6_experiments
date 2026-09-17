"""
Table 9 (Loquacious rows) and Table 10: hypothesis-count and search-RTF reduction from
dynamic pruning, plus the tuned dynamic-pruning parameters, for all three Loquacious models
on dev.all.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.loquacious import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_lexfree_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(
            model=models["ctc_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.all"]
        )
    )

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], train_corpus_key="train.medium", variants=variants, corpora=["dev.all"]
        )
    )

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_phoneme.run(
            model=models["ffnn_transducer_phoneme"], variants=variants, corpora=["dev.all"]
        )
    )

    return recog_results
