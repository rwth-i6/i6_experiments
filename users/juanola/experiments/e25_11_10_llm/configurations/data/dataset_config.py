from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass(frozen=True)
class DatasetConfig:
    """
    Dataset configuration base dataclass.

    Can contain default values.
    """

    preemphasis: Optional[float]
    peak_normalization: bool
    train_seq_ordering: str
    sampling_alpha: float = 0.7

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


"""
Specific configurations set below.
"""


def dataset_baseline() -> DatasetConfig:
    return DatasetConfig(
        preemphasis=None,
        peak_normalization=True,
        train_seq_ordering="laplace:.1000",
        sampling_alpha=0.7,
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
