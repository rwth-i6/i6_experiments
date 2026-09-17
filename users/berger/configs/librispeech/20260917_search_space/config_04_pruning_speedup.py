"""
Table IX (LibriSpeech rows) and Table X: hypothesis-count and search-RTF reduction from
dynamic pruning, plus the tuned dynamic-pruning parameters, for all three LibriSpeech models
on dev-other.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.librispeech import recognition
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
        variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
        variant.descriptor += "_gpu"
        variant.search_mode_params.gpu_mem_rqmt = 24
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
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True]:
        variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.search_algorithm_params.max_beam_sizes = [128]
        variant.search_mode_params.mem_rqmt = 32
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"], variants=variants, corpora=["dev-other"]))

    variants = []

    for score_threshold in [True, False]:
        variant = recognition.ctc_phoneme.default_offline_4gram_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ctc_phoneme.default_offline_trafo_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True]:
        variant = recognition.ctc_phoneme.default_offline_trafo_gpu_recog_variant()
        # variant.search_algorithm_params.max_beam_sizes = [128]
        variant.search_mode_params.mem_rqmt = 32
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    recog_results.extend(
        recognition.ctc_phoneme.run(model=models["ctc_phoneme"], variants=variants, corpora=["dev-other"])
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
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
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
            variant.search_algorithm_params.word_end_score_threshold = None
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
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True, False]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    for score_threshold in [True]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.search_algorithm_params.max_beam_sizes = [512]
        variant.search_mode_params.mem_rqmt = 32
        if score_threshold:
            variant.descriptor += "_dyn"
        else:
            variant.descriptor += "_fixed"
            variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.word_end_score_threshold = None
        variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    return recog_results
