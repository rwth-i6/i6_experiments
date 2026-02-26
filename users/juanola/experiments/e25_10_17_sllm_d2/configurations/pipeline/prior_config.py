from dataclasses import dataclass


@dataclass(frozen=True)
class PriorConfig:
    """
    Prior (inference) configuration base dataclass.

    Can contain default values.
    """

    batch_size: int

    forward_method: str

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


"""
Specific configurations set below.
"""


def prior_v1() -> PriorConfig:
    return PriorConfig(
        batch_size=16_000,
        forward_method="prior_step_v1"
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
