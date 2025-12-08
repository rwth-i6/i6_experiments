from dataclasses import dataclass, replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.data.dataset_config import DatasetConfig, \
    dataset_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.data.label_config import LabelConfig, \
    label_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.network.network_config import NetworkConfig, \
    network_baseline, network_SLLM_dropout, network_SLLM_tuned_dropout
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.pipeline.prior_config import \
    PriorConfig, prior_v1
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.pipeline.search_config import \
    SearchConfig, search_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.configs.pipeline.training_config import \
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
    prior: PriorConfig
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
        prior=prior_v1(),
        search=search_baseline(),
    )


def exp_v2() -> ExperimentConfig:
    return replace(exp_baseline(), network=network_SLLM_dropout())


def exp_v3() -> ExperimentConfig:
    return replace(exp_baseline(), network=network_SLLM_tuned_dropout())

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
