from ...model_pipelines.common.experiment_context import ExperimentContext
from ...model_pipelines.common.report import register_recog_report
from . import training
from . import recognition


def run_small() -> None:
    with ExperimentContext("training"):
        with ExperimentContext("bpe_ctc/baseline_model"):
            bpe_ctc_model = training.small.bpe_ctc.run()

        with ExperimentContext("bpe_ffnn_transducer/baseline_model"):
            bpe_ffnn_transducer_model = training.small.bpe_ffnn_transducer.run()

        with ExperimentContext("phoneme_ffnn_transducer/baseline_model"):
            phoneme_ffnn_transducer_model = training.small.phoneme_ffnn_transducer.run()

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_ctc/baseline_model/baseline_recog"):
            register_recog_report(recognition.bpe_ctc.run(bpe_ctc_model, train_corpus_key="train.small"))

        with ExperimentContext("bpe_ffnn_transducer/baseline_model/baseline_recog"):
            register_recog_report(
                recognition.bpe_ffnn_transducer.run(bpe_ffnn_transducer_model, train_corpus_key="train.small")
            )

        with ExperimentContext("phoneme_ffnn_transducer/baseline_model/baseline_recog"):
            register_recog_report(recognition.phoneme_ffnn_transducer.run(phoneme_ffnn_transducer_model))


def run_medium() -> None:
    with ExperimentContext("training"):
        with ExperimentContext("bpe_aed/baseline_model"):
            bpe_aed_model = training.medium.bpe_aed.run()

        with ExperimentContext("bpe_combination_model/baseline_model"):
            training.medium.bpe_combination_model.run()

        with ExperimentContext("bpe_ffnn_transducer/baseline_model"):
            bpe_ffnn_transducer_model = training.medium.bpe_ffnn_transducer.run()

        with ExperimentContext("bpe_phoneme_ctc/baseline_model"):
            training.medium.bpe_phoneme_ctc.run()

        with ExperimentContext("phoneme_ffnn_transducer/baseline_model"):
            phoneme_ffnn_transducer_model = training.medium.phoneme_ffnn_transducer.run()

    with ExperimentContext("recognition"):
        with ExperimentContext("bpe_aed/baseline_model/baseline_recog"):
            register_recog_report(recognition.bpe_aed.run(bpe_aed_model, train_corpus_key="train.medium"))

        with ExperimentContext("bpe_ffnn_transducer/baseline_model/baseline_recog"):
            register_recog_report(
                recognition.bpe_ffnn_transducer.run(bpe_ffnn_transducer_model, train_corpus_key="train.medium")
            )

        with ExperimentContext("phoneme_ffnn_transducer/baseline_model/baseline_recog"):
            register_recog_report(recognition.phoneme_ffnn_transducer.run(phoneme_ffnn_transducer_model))
