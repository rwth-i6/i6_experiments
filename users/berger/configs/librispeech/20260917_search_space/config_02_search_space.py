"""
Tables III and IV: WER, search errors and within-word hypothesis-count percentiles over the
pruning parameters, for BPE CTC (Table III) and BPE transducer (Table IV) lexicon-constrained
search with the word-level Transformer LM on dev-other.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.librispeech import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []
    for score_threshold, max_beam_size in [
        (0.0, 1),
        (2.0, 16),
        (4.0, 16),
        (8.0, 128),
        (12.0, 256),
        (16.0, 256),
        (16.0, 1024),
    ]:
        variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        variant.search_algorithm_params.score_thresholds = [score_threshold]
        variant.compute_search_errors = True
        variants.append(variant)

        # variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        # variant.search_algorithm_params.score_thresholds = [score_threshold]
        # variant.compute_search_errors = True
        # variants.append(variant)
    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"], variants=variants, corpora=["dev-other"]))

    # Table 4
    variants = []
    for score_threshold, max_beam_size in [(0.0, 1), (2.0, 16), (4.0, 16), (8.0, 128), (12.0, 256), (16.0, 1024)]:
        variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
        variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        variant.search_algorithm_params.score_thresholds = [score_threshold]
        variant.compute_search_errors = True
        variants.append(variant)

        # variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
        # variant.descriptor += f"_score-{score_threshold}_beam-{max_beam_size}_search_err"
        # variant.search_algorithm_params.max_beam_sizes = [max_beam_size]
        # variant.search_algorithm_params.score_thresholds = [score_threshold]
        # variant.compute_search_errors = True
        # variants.append(variant)
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other"]
        )
    )

    return recog_results
