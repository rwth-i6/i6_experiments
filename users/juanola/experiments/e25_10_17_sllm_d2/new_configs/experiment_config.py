from dataclasses import dataclass, replace

from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.dataset_config import DatasetConfig, dataset_baseline
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.network.network_config import NetworkConfig, network_v1, \
    network_v2
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.prior_config import PriorConfig, prior_v1
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.search_config import SearchConfig, search_baseline
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.training_config import TrainingConfig, training_baseline


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Experiment configuration base dataclass. Contains all parameters needed for the experiment.

    Can contain default values.
    """
    # general: GeneralParams # add if needed
    dataset: DatasetConfig

    network: NetworkConfig

    training: TrainingConfig
    prior: PriorConfig
    search: SearchConfig


"""
Specific configurations set below.
"""


def exp_v1() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        network=network_v1(),
        training=training_baseline(),
        prior=prior_v1(),
        search=search_baseline(),
    )


def exp_v2() -> ExperimentConfig:
    return replace(exp_v1, network=network_v2())

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
