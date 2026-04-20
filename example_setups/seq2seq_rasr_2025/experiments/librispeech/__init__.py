from typing import Dict, List, Optional, Tuple

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from ...model_pipelines.common.train import TrainedModel
from . import recognition, training


def run_all(report_filename: Optional[str] = None) -> Tuple[Dict[str, TrainedModel], List[RecogResult]]:
    models = {
        "aed_bpe": training.aed_bpe.run(descriptor="aed_bpe"),
        "ctc_bpe": training.ctc_bpe.run(descriptor="ctc_bpe"),
        "ctc_phoneme": training.ctc_phoneme.run(descriptor="ctc_phoneme"),
        "ffnn_transducer_bpe": training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "ffnn_transducer_pruned_bpe": training.ffnn_transducer_pruned_bpe.run(descriptor="ffnn_transducer_pruned_bpe"),
        "full_ctx_transducer_bpe": training.full_ctx_transducer_bpe.run(descriptor="full_ctx_transducer_bpe"),
    }

    recog_results = []
    recog_results.extend(recognition.aed_bpe.run(model=models["aed_bpe"]))
    recog_results.extend(recognition.ctc_bpe.run(model=models["ctc_bpe"]))
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_bpe"]))
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=models["ffnn_transducer_pruned_bpe"]))
    recog_results.extend(recognition.full_ctx_transducer_bpe.run(model=models["full_ctx_transducer_bpe"]))
    recog_results.extend(recognition.ctc_phoneme.run(model=models["ctc_phoneme"]))
    recog_results.extend(recognition.aed_ctc_bpe.run(aed_model=models["aed_bpe"], ctc_model=models["ctc_bpe"]))

    if report_filename is not None:
        register_recog_report(recog_results, filename=report_filename)
    return models, recog_results
