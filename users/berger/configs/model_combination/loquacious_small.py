"""Standalone Loquacious-small config; preserves the combined config's aliases."""


def py() -> None:
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import (
        ExperimentContext,
    )
    from .loquacious import run_small

    with ExperimentContext("loquacious_small"):
        run_small()


def main() -> None:
    py()
