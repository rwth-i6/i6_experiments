"""Representative Loquacious-medium examples; opt-in due to dataset size."""


def py():
    from i6_experiments.example_setups.seq2seq_rasr_2025.experiments.loquacious import run_medium
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import (
        ExperimentContext,
    )

    with ExperimentContext("loquacious_medium"):
        return run_medium(report_filename="loquacious-medium.csv")


def main():
    return py()
