from i6_experiments.example_setups.seq2seq_rasr_2025.model_pipelines.common.experiment_context import ExperimentContext

from .librispeech import run as run_librispeech
from .loquacious import run_small as run_loquacious_small
from .loquacious import run_medium as run_loquacious_medium


def main() -> None:
    with ExperimentContext("librispeech"):
        run_librispeech()

    with ExperimentContext("loquacious_small"):
        run_loquacious_small()

    with ExperimentContext("loquacious_medium"):
        run_loquacious_medium()
