from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Dataset configuration base dataclass.

    Can contain default values.
    """
    # TODO: this only for example
    epochs: int


"""
Specific configurations set below.
"""


def get_dataset_config_v1() -> TrainingConfig:
    return TrainingConfig()
