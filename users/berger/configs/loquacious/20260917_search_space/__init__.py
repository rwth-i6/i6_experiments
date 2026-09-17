"""
Loquacious experiments for "A Comparison of Search Spaces and Search Strategies for
Attention-free ASR" (SLT 2026).

Each `config_*` module collects the recognition calls behind one artifact of the paper; the
module docstrings name the table or figure.

The prerequisites live in `i6_experiments.users.berger.seq2seq_rasr_2025`.
"""

__all__ = ["main"]

from typing import List

from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.recog import RecogResult
from i6_experiments.users.berger.seq2seq_rasr_2025.model_pipelines.common.report import register_recog_report

from . import (
    config_01_search_strategies,
    config_02_search_space,
    config_03_pruning_speedup,
    config_04_rtf_vs_wer_curves,
    models,
)


def main() -> None:
    with ExperimentContext("20260917_search_space/loquacious"):
        trained_models = models.run()

        recog_results: List[RecogResult] = []
        for config in [
            config_01_search_strategies,
            config_02_search_space,
            config_03_pruning_speedup,
            config_04_rtf_vs_wer_curves,
        ]:
            recog_results.extend(config.run(trained_models))

        register_recog_report(recog_results)
