from typing import Dict, Union, Optional, Any
from dataclasses import dataclass

from i6_experiments.common.setups.returnn.datasets import Dataset
from i6_experiments.common.setups.returnn.datastreams.base import Datastream

from sisyphus import tk


@dataclass(frozen=True)
class TrainingDatasets:
    train: Dataset
    eval_datasets: Dict[str, Dataset]
    datastreams: Dict[str, Datastream]


class LabelDatastreamWoVocab(Datastream):
    """
    Defines a datastream for labels represented by indices using the default `Vocabulary` class of RETURNN

    This defines a word-(unit)-based vocabulary
    """

    def __init__(
        self,
        available_for_inference: bool,
        vocab_size: Union[tk.Variable, int],
    ):
        """

        :param available_for_inference:
        :param vocab: word vocab file path (pickle containing dictionary)
        :param vocab_size: used for the actual dimension
        :param unk_label: unknown label
        """
        super().__init__(available_for_inference)
        self.vocab_size = vocab_size

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None, **kwargs) -> Dict[str, Any]:
        """
        :param tk.Variable|int vocab_size: number of labels
        :rtype: dict[str]
        """
        d = {
            **super().as_returnn_extern_data_opts(available_for_inference=available_for_inference),
            "shape": (None,),
            "dim": self.vocab_size,
            "sparse": True,
        }
        d.update(kwargs)
        return d


@dataclass()
class DatasetSettings:
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

    # training settings
    train_partition_epoch: int
    train_seq_ordering: Optional[str]

    # multiproc settings
    num_workers: int = 2
