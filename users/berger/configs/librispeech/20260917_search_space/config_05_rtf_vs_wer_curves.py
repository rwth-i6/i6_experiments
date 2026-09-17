"""
Figures 2 and 3: search RTF vs. WER curves for fixed and dynamic pruning with the BPE
transducer, using the subword-level LSTM LM (Figure 2) and the word-level Transformer LM
(Figure 3) on dev-other.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.librispeech import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []

    for score_threshold in [None, 2.0, 4.0, 6.0]:
        for max_beam_size in [4, 8, 16, 32, 64, 128]:
            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold, score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)

    for score_threshold in [None, 2.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
        for max_beam_size in [4, 8, 16, 32, 64, 128]:
            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_gpu"
            variant.search_mode_params.gpu_mem_rqmt = 24
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold, score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)

    for score_threshold in [None, 4.0, 8.0, 12.0, 16.0]:
        for max_beam_size in [16, 64, 256, 512, 1024]:
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            variant.search_mode_params.mem_rqmt = 32
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
                variant.search_algorithm_params.word_end_score_threshold = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

    for score_threshold in [None, 4.0, 8.0, 10.0, 12.0, 14.0, 16.0]:
        for max_beam_size in [16, 32, 64, 128, 256, 512, 1024]:
            if not score_threshold and max_beam_size >= 128:
                continue
            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
            variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}"
            variant.search_mode_params.mem_rqmt = 32
            if score_threshold is None:
                variant.search_algorithm_params.score_thresholds = None
                variant.search_algorithm_params.word_end_score_threshold = None
            else:
                variant.search_algorithm_params.score_thresholds = [score_threshold]
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    return recog_results
