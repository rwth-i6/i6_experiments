"""Small and medium Loquacious template recipes."""

from typing import Optional
from ...model_pipelines.common.report import register_recog_report
from . import recognition, training


def _run(size, report_filename):
    names = ["ctc_bpe", "ffnn_transducer_bpe"]
    if size == "medium":
        names.append("aed_bpe")
    models = {name: getattr(getattr(training, size), name).run(descriptor=name) for name in names}
    results = []
    for name, model in models.items():
        results.extend(getattr(recognition, name).run(model=model, train_corpus_key="train." + size))
    if report_filename is not None:
        register_recog_report(results, filename=report_filename)
    return models, results


def run_small(report_filename: Optional[str] = None):
    return _run("small", report_filename)


def run_medium(report_filename: Optional[str] = None):
    return _run("medium", report_filename)
