from ....model_pipelines.common.experiment_context import ExperimentContext
from ....model_pipelines.common.report import register_recog_report
from . import (
    bpe_aed,
    bpe_combination_model,
    bpe_ffnn_transducer,
    bpe_phoneme_ctc,
    phoneme_ffnn_transducer,
)


def run_all() -> None:
    with ExperimentContext("bpe_ffnn_transducer/baseline"):
        register_recog_report(bpe_ffnn_transducer.run_all())
    with ExperimentContext("phoneme_ffnn_transducer/baseline"):
        register_recog_report(phoneme_ffnn_transducer.run_all())
    with ExperimentContext("bpe_aed/baseline"):
        register_recog_report(bpe_aed.run_all())
    with ExperimentContext("bpe_phoneme_ctc/baseline"):
        bpe_phoneme_ctc.run_all()
    with ExperimentContext("bpe_combination_model/baseline"):
        bpe_combination_model.run_all()
