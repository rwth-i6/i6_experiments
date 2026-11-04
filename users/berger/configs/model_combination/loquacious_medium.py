"""Standalone Loquacious-medium config; respects existing disabled experiments."""


def py() -> None:
    from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import (
        ExperimentContext,
    )
    from .loquacious import run_medium

    with ExperimentContext("loquacious_medium"):
        run_medium()


def main() -> None:
    py()
