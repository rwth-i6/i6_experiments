"""
Table 1: WER and search RTF on LibriSpeech for BPE CTC, phoneme CTC and BPE transducer
under lexicon-free and lexicon-constrained search with the different LM integrations.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.librispeech import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []
    variant = recognition.ctc_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_lexfree_trafo_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_trafo_recog_variant_gpu()
    variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(model=models["ctc_bpe"], variants=variants, corpora=["dev-other", "test-other"])
    )

    variants = []
    variant = recognition.ctc_phoneme.default_offline_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_phoneme.default_offline_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_phoneme.default_offline_trafo_gpu_recog_variant()
    variants.append(variant)

    recog_results.extend(
        recognition.ctc_phoneme.run(model=models["ctc_phoneme"], variants=variants, corpora=["dev-other", "test-other"])
    )

    variants = []

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_bpe_trafo_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_lstm_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_tree_trafo_recog_variant_gpu()
    variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"], variants=variants, corpora=["dev-other", "test-other"]
        )
    )

    return recog_results
