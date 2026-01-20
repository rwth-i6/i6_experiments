from dataclasses import dataclass

from .data.dataset_config import DatasetConfig, dataset_baseline_train_corpus_text, dataset_baseline_all_data
from .data.label_config import LabelConfig, label_baseline
from .network.network_config import NetworkConfig, network_base, network_small
from .pipeline.search_config import SearchConfig, search_baseline_v2
from .pipeline.training_config import TrainingConfig, training_transcript_data, training_baseline_test, training_lm_data


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
        dataset=dataset_baseline_train_corpus_text(),
        labels=label_baseline(),
        network=network_base(),
        training=training_transcript_data(),
        search=search_baseline_v2(),
    )


def exp_1_1() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline_train_corpus_text(),
        labels=label_baseline(),
        network=network_base(),
        training=training_transcript_data(),
        search=search_baseline_v2(),
    )


def exp_1_2() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline_all_data(),
        labels=label_baseline(),
        network=network_base(),
        training=training_lm_data(),
        search=search_baseline_v2(),
    )


def exp_1_3() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline_train_corpus_text(),
        labels=label_baseline(),
        network=network_small(),
        training=training_transcript_data(),
        search=search_baseline_v2(),
    )


def exp_1_4() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline_all_data(),
        labels=label_baseline(),
        network=network_small(),
        training=training_lm_data(),
        search=search_baseline_v2(),
    )


def exp_baseline_test() -> ExperimentConfig:
    return ExperimentConfig(
        dataset=dataset_baseline_train_corpus_text(),
        labels=label_baseline(),
        network=network_base(),
        training=training_baseline_test(),
        search=search_baseline_v2(),
    )

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
