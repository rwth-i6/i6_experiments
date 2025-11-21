from dataclasses import dataclass


@dataclass
class EncoderConfig:
    """
    Dataset configuration base dataclass.

    Can contain default values.
    """
    # TODO: this only for example
    n_layers: int = 1


"""
Specific configurations set below.
"""


def get_dataset_config_v1() -> EncoderConfig:
    return EncoderConfig()
