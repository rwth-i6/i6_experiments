"""Representative Loquacious-small CTC and context-1 transducer examples."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_small
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import (
        ExperimentContext,
    )

    with ExperimentContext("loquacious_small"):
        return run_small(report_filename="loquacious-small.csv")


def main():
    return py()
