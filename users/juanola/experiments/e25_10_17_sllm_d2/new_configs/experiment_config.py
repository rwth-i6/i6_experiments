from dataclasses import dataclass

from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.dataset_config import DatasetConfig, get_dataset_config_v1
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.prior_config import PriorConfig
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.search_config import SearchConfig
from users.juanola.experiments.e25_10_17_sllm_d2.new_configs.training_config import TrainingConfig


# TODO: add names for configs
# TODO: inheritance from some root config to force name, or other properties?

@dataclass
class ExperimentConfig:
    """
    Experiment configuration base dataclass. Contains all parameters needed for the experiment.

    Can contain default values.
    """
    general_config = None
    dataset_config: DatasetConfig = None
    training_config: TrainingConfig = None
    prior_config: PriorConfig = None
    search_config: SearchConfig = None


"""
Specific configurations set below.
"""


def get_exp_config_v1() -> ExperimentConfig:
    return ExperimentConfig(
        general_config=None,
        dataset_config=get_dataset_config_v1(),
        training_config=None,
        prior_config=None,
        search_config=None,
    )


def get_exp_config_v2() -> ExperimentConfig:
    return ExperimentConfig(
        general_config=None,
        dataset_config=None,
        training_config=None,
        prior_config=None,
        search_config=None,
    )
