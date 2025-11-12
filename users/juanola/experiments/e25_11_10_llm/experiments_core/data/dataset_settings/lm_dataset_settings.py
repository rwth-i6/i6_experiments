from dataclasses import dataclass


@dataclass()
class LMDatasetSettings:
    train_partition_epoch: int
    train_seq_ordering: str