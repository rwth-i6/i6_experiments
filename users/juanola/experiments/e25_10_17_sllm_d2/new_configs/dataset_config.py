from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """
    Dataset configuration base dataclass.

    Can contain default values.
    """
    # TODO: this only for example
    preemphasis = None
    peak_normalization: bool
    train_partition_epoch: int
    train_seq_ordering = "laplace:.1000"
    train_additional_options = {
        "epoch_wise_filter": {(1, 5): {"max_mean_len": 1000}}
    },


"""
Specific configurations set below.
"""


def get_dataset_config_v1() -> DatasetConfig:
    return DatasetConfig()
