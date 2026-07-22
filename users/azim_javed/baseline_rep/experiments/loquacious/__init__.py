from typing import Dict, List, Optional, Tuple

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from ...model_pipelines.common.train import TrainedModel
from . import recognition, training


def run_small(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "ctc_bpe": training.small.ctc_bpe.run(descriptor="ctc_bpe"),
        "ffnn_transducer_bpe": training.small.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_phoneme": training.small.ffnn_transducer_phoneme.run(descriptor="ffnn_transducer_phoneme"),
    }

    recog_results = []
    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"], train_corpus_key="train.small"))
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_bpe"], train_corpus_key="train.small")
    )
    recog_results.extend(recognition.ffnn_transducer_phoneme.run(model=models["ffnn_transducer_phoneme"]))

    if report_filename is not None:
        register_recog_report(recog_results, filename=report_filename)
    return models, recog_results


def run_medium(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "aed_bpe": training.medium.aed_bpe.run(descriptor="aed_bpe"),
        "combination_model_bpe": training.medium.combination_model_bpe.run(descriptor="combination_model_bpe"),
        "ctc_bpe_phoneme": training.medium.ctc_bpe_phoneme.run(descriptor="ctc_bpe_phoneme"),
        "ffnn_transducer_bpe": training.medium.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_phoneme": training.medium.ffnn_transducer_phoneme.run(descriptor="ffnn_transducer_phoneme"),
    }

    recog_results = []
    recog_results.extend(recognition.aed_bpe.run(model=models["aed_bpe"], train_corpus_key="train.medium"))
    recog_results.extend(
        recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_bpe"], train_corpus_key="train.medium")
    )
    recog_results.extend(recognition.ffnn_transducer_phoneme.run(model=models["ffnn_transducer_phoneme"]))

    if report_filename is not None:
        register_recog_report(recog_results, filename=report_filename)

    return models, recog_results
