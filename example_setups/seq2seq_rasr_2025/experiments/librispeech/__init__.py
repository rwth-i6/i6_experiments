"""Representative LibriSpeech models; custom sweeps belong in user setups."""

from typing import Optional
from ...model_pipelines.common.report import register_recog_report
from . import recognition, training


def run_all(report_filename: Optional[str] = None):
    models = {
        "ctc_bpe": training.ctc_bpe.run(descriptor="ctc_bpe"),
        "ffnn_transducer_bpe": training.ffnn_transducer_bpe.run(descriptor="ffnn_transducer_bpe"),
        "full_ctx_transducer_bpe": training.full_ctx_transducer_bpe.run(descriptor="full_ctx_transducer_bpe"),
        "aed_bpe": training.aed_bpe.run(descriptor="aed_bpe"),
    }
    results = []
    for name, model in models.items():
        results.extend(getattr(recognition, name).run(model=model))
    if report_filename is not None:
        register_recog_report(results, filename=report_filename)
    return models, results
