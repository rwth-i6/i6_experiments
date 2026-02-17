from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.report import register_recog_report
from . import training
from . import recognition


def run_all() -> None:
    with ExperimentContext("training"):
        with ExperimentContext("bpe_aed/baseline_model"):
            aed_model = training.bpe_aed.run()

        with ExperimentContext("bpe_ctc/baseline_model"):
            bpe_ctc_model = training.bpe_ctc.run()

        with ExperimentContext("bpe_ffnn_transducer/baseline_model"):
            transducer_model = training.bpe_ffnn_transducer.run()

        with ExperimentContext("phoneme_ctc/baseline_model"):
            phoneme_ctc_model = training.phoneme_ctc.run()

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_aed/baseline_model/baseline_recog"):
            register_recog_report(recognition.bpe_aed.run(aed_model))

        with ExperimentContext("bpe_ctc/baseline_model/baseline_recog"):
            register_recog_report(recognition.bpe_ctc.run(bpe_ctc_model))

        with ExperimentContext("bpe_ffnn_transducer/baseline_model/baseline_recog"):
            register_recog_report(recognition.bpe_ffnn_transducer.run(transducer_model))

        with ExperimentContext("phoneme_ctc/baseline_model/baseline_recog"):
            register_recog_report(recognition.phoneme_ctc.run(phoneme_ctc_model))

        with ExperimentContext("bpe_aed__bpe_ctc/baseline_model/baseline_recog"):
            register_recog_report(recognition.bpe_aed__bpe_ctc.run(aed_model=aed_model, ctc_model=bpe_ctc_model))
