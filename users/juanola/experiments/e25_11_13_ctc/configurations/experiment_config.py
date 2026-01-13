from dataclasses import dataclass

from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.data.dataset_config import DatasetConfig, \
    dataset_baseline
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.data.label_config import LabelConfig, \
    label_baseline
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.network.network_config import NetworkConfig, \
    network_baseline
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.pipeline.search_config import SearchConfig, \
    search_ctc_greedy
from i6_experiments.users.juanola.experiments.e25_11_13_ctc.configurations.pipeline.training_config import \
    TrainingConfig, training_baseline


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Experiment configuration base dataclass. Contains all parameters needed for the experiment.

    Can contain default values.
    """

    # general: GeneralParams # add if needed
    dataset: DatasetConfig
    labels: LabelConfig

    network: NetworkConfig

    training: TrainingConfig
    search: SearchConfig


"""
Specific configurations set below.
"""


def exp_baseline() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_baseline(),
        training=training_baseline(),
        search=search_ctc_greedy(),
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
