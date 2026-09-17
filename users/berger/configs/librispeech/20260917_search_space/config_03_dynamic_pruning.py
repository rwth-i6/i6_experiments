"""
Tables 7 and 8: fixed vs. dynamic pruning broken down over the max beam size B_ASR, for the
BPE transducer with the subword-level LSTM LM (Table 7) and the word-level Transformer LM
(Table 8) on dev-other.
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
        for max_beam_size in [4, 16, 64, 256]:
            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_beam-{max_beam_size}"
            if score_threshold:
                variant.descriptor += "_dyn"
            else:
                variant.descriptor += "_fixed"
                variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)

            variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
            variant.descriptor += f"_beam-{max_beam_size}_gpu"
            variant.search_mode_params.gpu_mem_rqmt = 24
            if score_threshold:
                variant.descriptor += "_dyn"
            else:
                variant.descriptor += "_fixed"
                variant.search_algorithm_params.score_thresholds = None
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size, max_beam_size]
            variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    # Table 8
    variants = []
    for score_threshold in [True, False]:
        for max_beam_size in [4, 16, 64, 128, 512, 1024]:
            # variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
            # variant.descriptor += f"_beam-{max_beam_size}"
            # variant.search_mode_params.gpu_mem_rqmt = 24
            # if score_threshold:
            #     variant.descriptor += "_dyn"
            # else:
            #     variant.descriptor += "_fixed"
            #     variant.search_algorithm_params.score_thresholds = None
            #     variant.search_algorithm_params.word_end_score_threshold = None
            # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            # variants.append(variant)
            if not score_threshold and max_beam_size >= 128:
                continue

            variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
            variant.descriptor += f"_beam-{max_beam_size}"
            variant.search_mode_params.gpu_mem_rqmt = 24
            variant.search_mode_params.mem_rqmt = 32
            if score_threshold:
                variant.descriptor += "_dyn"
            else:
                variant.descriptor += "_fixed"
                variant.search_algorithm_params.score_thresholds = None
                variant.search_algorithm_params.word_end_score_threshold = None
            variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
            variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    return recog_results
