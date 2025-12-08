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
    train_additional_options: Optional[Dict[str, Any]]
    sampling_alpha: float = 0.7
    vocab_size: int = 10_240


"""
Specific configurations set below.
"""


def dataset_baseline() -> DatasetConfig:
    return DatasetConfig(
        preemphasis=None,
        peak_normalization=True,
        train_seq_ordering="laplace:.1000",
        train_additional_options={"epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}},
        sampling_alpha=0.7,
        vocab_size=10_240,
    )


# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
