import warnings
from dataclasses import dataclass, replace

from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.data.dataset_config import DatasetConfig, \
    dataset_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.data.label_config import LabelConfig, \
    label_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.network.network_config import \
    NetworkConfig, \
    network_baseline, network_SLLM_dropout, network_SLLM_tuned_dropout, network_SLLM_small_decoder, \
    network_linear_adapter, \
    network_SLLM_tuned_dropout_v2
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.prior_config import \
    PriorConfig, prior_v1
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.search_config import \
    SearchConfig, search_baseline
from i6_experiments.users.juanola.experiments.e25_10_17_sllm_d2.configurations.pipeline.training_config import \
    TrainingConfig, training_baseline, itc_batch_size_80k, itc_batch_size_150k, bsv2_lrv2, bsv2_lrv3


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
    warnings.warn(
        "[BUG] Doesn't use DROPOUT DROPOUT + intermediate_size too large",
        DeprecationWarning,
        stacklevel=2,
    )
    return replace(exp_baseline(), network=network_SLLM_tuned_dropout())

def exp_v4() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),

        network=network_SLLM_small_decoder(), # !!

        training=itc_batch_size_80k(),  # !!
        prior=prior_v1(),
        search=search_baseline(),
    )

def exp_v5() -> ExperimentConfig:
    return replace(exp_v4(), network=network_linear_adapter())

def exp_v6() -> ExperimentConfig:
    return replace(exp_v4(), training=itc_batch_size_150k())


def exp_v7() -> ExperimentConfig:
    """
    V4 but with proper decoder parameters
    :return:
    """
    return ExperimentConfig(
        dataset=dataset_baseline(),
        labels=label_baseline(),

        network=network_SLLM_tuned_dropout_v2(), # !!

        training=itc_batch_size_80k(),
        prior=prior_v1(),
        search=search_baseline(),
    )

def exp_v8_1() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv2())

def exp_v8_2() -> ExperimentConfig:
    return replace(exp_v7(), training=bsv2_lrv3())

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
