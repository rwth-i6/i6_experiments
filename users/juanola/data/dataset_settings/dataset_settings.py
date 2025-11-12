from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass()
class ReturnnDatasetSettings:
    """
    A helper structure for the dataset settings that are configurable in RETURNN

    Args:
        custom_prcessing_function: the name of a python function added to the config
            this function can be used to process the input waveform
        partition_epoch: use this split of the data for one training epoch
        epoch_wise_filters: can be used to limit e.g. the sequence lengths at the beginning of the training
        seq_ordering: see RETURNN settings on sequence sorting
        preemphasis: filter scale for high-pass z-filter
        peak_normalization: normalize input utterance to unit amplitude peak
    """

    # general settings
    preemphasis: Optional[float]
    peak_normalization: bool

    # training settings
    train_partition_epoch: int
    train_seq_ordering: str
    train_additional_options: Optional[Dict[str, Any]] = None