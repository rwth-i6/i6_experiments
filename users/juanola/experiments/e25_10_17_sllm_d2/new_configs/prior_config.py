from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    """
    Prior (inference) configuration base dataclass.

    Can contain default values.
    """

    batch_size: int


"""
Specific configurations set below.
"""


def prior_v1() -> PriorConfig:
    return PriorConfig(batch_size=16_000)


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
