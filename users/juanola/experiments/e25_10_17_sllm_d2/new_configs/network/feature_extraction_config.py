from dataclasses import dataclass


@dataclass
class FeatureExtractionConfig:
    """
    Dataset configuration base dataclass.

    Can contain default values.
    """
    # TODO: this only for example
    whatever: int = 1


"""
Specific configurations set below.
"""


def get_dataset_config_v1() -> FeatureExtractionConfig:
    return FeatureExtractionConfig()
