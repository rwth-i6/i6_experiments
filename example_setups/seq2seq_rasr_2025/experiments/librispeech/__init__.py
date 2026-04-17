from typing import List

from ...model_pipelines.common.recog import RecogResult
from ...model_pipelines.common.report import register_recog_report
from . import recognition, training


def run_all(register_report: bool = True) -> List[RecogResult]:
    aed_model = training.aed_bpe.run(descriptor="aed_bpe")
    ctc_bpe_model = training.ctc_bpe.run(descriptor="ctc_bpe")
    ctc_phoneme_model = training.ctc_phoneme.run(descriptor="ctc_phoneme")
    transducer_model = training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe")
    transducer_model_pruned = training.ffnn_transducer_pruned_bpe.run(descriptor="ffnn_transducer_pruned_bpe")
    full_ctx_transducer_model = training.full_ctx_transducer_bpe.run(descriptor="full_ctx_transducer_bpe")

    recog_results = []
    recog_results.extend(recognition.aed_bpe.run(model=aed_model))
    recog_results.extend(recognition.ctc_bpe.run(model=ctc_bpe_model))
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=transducer_model))
    recog_results.extend(recognition.ffnn_transducer_bpe.run(model=transducer_model_pruned))
    recog_results.extend(recognition.full_ctx_transducer_bpe.run(model=full_ctx_transducer_model))
    recog_results.extend(recognition.ctc_phoneme.run(model=ctc_phoneme_model))
    recog_results.extend(recognition.aed_ctc_bpe.run(aed_model=aed_model, ctc_model=ctc_bpe_model))

    if register_report:
        register_recog_report(recog_results, filename="baseline_report.txt")
    return recog_results
