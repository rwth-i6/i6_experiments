from dataclasses import dataclass

from .data.dataset_config import DatasetConfig, dataset_baseline
from .data.label_config import LabelConfig, label_baseline
from .network.network_config import NetworkConfig, network_baseline
from .pipeline.search_config import SearchConfig, search_baseline_v2
from .pipeline.training_config import TrainingConfig, training_baseline, training_baseline_test


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
        search=search_baseline_v2(),
    )

def exp_baseline_test() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),
        network=network_baseline(),
        training=training_baseline_test(),
        search=search_baseline_v2(),
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
