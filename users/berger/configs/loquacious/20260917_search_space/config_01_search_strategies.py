"""
Table 2: WER and search RTF on Loquacious for BPE CTC, BPE transducer and phoneme
transducer under lexicon-free and lexicon-constrained search with the word-level 4-gram LM.
"""

__all__ = ["run"]

from typing import Dict, List

from i6_experiments.users.berger.seq2seq_rasr_2025.experiments.loquacious import recognition
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.train import TrainedModel


def run(models: Dict[str, TrainedModel]) -> List[RecogResult]:
    recog_results: List[RecogResult] = []

    variants = []
    variant = recognition.ctc_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_recog_variant()
    variants.append(variant)

    variant = recognition.ctc_bpe.default_offline_tree_4gram_recog_variant()
    variants.append(variant)

    recog_results.extend(
        recognition.ctc_bpe.run(
            model=models["ctc_bpe"],
            train_corpus_key="train.medium",
            variants=variants,
            corpora=["dev.all", "test.all"],
        )
    )

    variants = []
    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_bpe.default_offline_lexfree_recog_variant()
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

    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(
            model=models["ffnn_transducer_bpe"],
            train_corpus_key="train.medium",
            variants=variants,
            corpora=["dev.all", "test.all"],
        )
    )

    variants = []
    variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
    variants.append(variant)

    variant = recognition.ffnn_transducer_phoneme.default_offline_4gram_recog_variant()
    variant.descriptor += "_gpu"
    variant.search_mode_params.gpu_mem_rqmt = 24
    variants.append(variant)

    recog_results.extend(
        recognition.ffnn_transducer_phoneme.run(
            model=models["ffnn_transducer_phoneme"],
            variants=variants,
            corpora=["dev.all", "test.all"],
        )
    )

    return recog_results
