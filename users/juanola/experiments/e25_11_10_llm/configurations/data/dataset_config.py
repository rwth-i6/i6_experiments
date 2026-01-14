import dataclasses
from dataclasses import dataclass
from typing import Optional


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

    # Which data to use - by default only train corpus text
    use_train_corpus_text: bool = True
    use_normalized_lm_data: bool = False

    def __post_init__(self):
        """
        Assertions for parameters.
        """
        pass


"""
Specific configurations set below.
"""


def dataset_baseline_train_corpus_text() -> DatasetConfig:
    return DatasetConfig(
        preemphasis=None,
        peak_normalization=True,
        train_seq_ordering="laplace:.1000",
        sampling_alpha=0.7,
        use_train_corpus_text = True,
        use_normalized_lm_data = False
    )

def dataset_baseline_normalized_lm_data() -> DatasetConfig:
    return dataclasses.replace(dataset_baseline_train_corpus_text(),
                               use_train_corpus_text = False,
                               use_normalized_lm_data = True)

def dataset_baseline_all_data() -> DatasetConfig:
    return dataclasses.replace(dataset_baseline_train_corpus_text(),
                               use_train_corpus_text = True,
                               use_normalized_lm_data = True)

# For inheritance use: dataclasses.replace(OriginalClass, elements_to_modify)
