"""Run the representative LibriSpeech template models and LMs."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.librispeech import run_all
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import (
        ExperimentContext,
    )

    with ExperimentContext("librispeech"):
        return run_all(report_filename="librispeech.csv")


def main():
    return py()
