from ....model_pipelines.common.experiment_context import ExperimentContext
from ....model_pipelines.common.report import register_recog_report
from . import (
    bpe_ctc,
    bpe_ffnn_transducer,
    phoneme_ffnn_transducer,
)


def run_all() -> None:
    with ExperimentContext("bpe_ctc/baseline"):
        register_recog_report(bpe_ctc.run_all())
    with ExperimentContext("bpe_ffnn_transducer/baseline"):
        register_recog_report(bpe_ffnn_transducer.run_all())
    with ExperimentContext("phoneme_ffnn_transducer/baseline"):
        register_recog_report(phoneme_ffnn_transducer.run_all())
