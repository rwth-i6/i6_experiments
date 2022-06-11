"""
Helper classes around RETURNN datasets
"""
__all__ = ["GenericDataset", "ControlDataset", "MetaDataset"]

import abc
from sisyphus import tk
from typing import *


class GenericDataset(abc.ABC):
    """
    Basic interface to create parameter dictionaries for all datasets inheriting from `returnn.datasets.basic.Dataset`
    """

    def __init__(self, *, other_opts: Optional[Dict]):
        """
        :param other_opts: any options not explicitely covered by the helper classes or for debugging purposes
        """
        self.other_opts = other_opts

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        :return: data dict for SprintDataset, OggZipDataset, etc
        """
        return self.other_opts or {}


class ControlDataset(GenericDataset, abc.ABC):
    """
    A template for all Datasets that allow for data control (sequence ordering, partitioning, subsets)
    """

    def __init__(
        self,
        *,
        partition_epoch: Optional[int] = None,
        seq_list_filter_file: Optional[tk.Path] = None,
        seq_ordering: Optional[str] = None,
        # super parameters
        other_opts: Optional[Dict] = None,
    ):
        """
        :param partition_epoch: partition the data into N parts
        :param seq_list_filter_file: text file (gzip/plain) or pkl containg list of sequence tags to use
        :param seq_ordering: see `https://returnn.readthedocs.io/en/latest/dataset_reference/index.html`_.
        :param other_opts: custom options directly passed to the dataset
        """
        super().__init__(other_opts=other_opts)
        self.partition_epoch = partition_epoch or 1
        self.seq_list_filter_file = seq_list_filter_file
        self.seq_ordering = seq_ordering

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "partition_epoch": self.partition_epoch,
        }
        if self.seq_ordering:
            d["seq_ordering"] = self.seq_ordering
        if self.seq_list_filter_file:
            d["seq_list_filter_file"] = self.seq_list_filter_file

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d


class MetaDataset(GenericDataset):
    """
    Represents :class:`MetaDataset` in RETURNN

    Only allows the MetaDataset to be used with an explicit control dataset.
    """

    def __init__(
        self,
        *,
        data_map: Dict[str, Tuple[str, str]],
        datasets: Dict[str, Union[Dict, GenericDataset]],
        seq_order_control_dataset: str,
        other_opts: Optional[Dict[str, Any]] = None,
    ):
        """
        :param data_map:
        :param datasets:
        :param seq_order_control_dataset:
        :param dict other_opts:
        """
        super().__init__(other_opts=other_opts)
        self.data_map = data_map
        self.datasets = {
            k: v if isinstance(v, dict) else v.as_returnn_opts()
            for k, v in datasets.items()
        }
        assert seq_order_control_dataset in datasets
        self.seq_order_control_dataset = seq_order_control_dataset

    def as_returnn_opts(self):
        d = {
            "class": "MetaDataset",
            "data_map": self.data_map,
            "datasets": self.datasets,
            "seq_order_control_dataset": self.seq_order_control_dataset,
        }

        sd = super().as_returnn_opts()
        assert all([k not in sd.keys() for k in d.keys()]), (
            "conflicting keys in %s and %s"
            % (str(list(sd.keys())), str(list(d.keys()))),
        )
        d.update(sd)

        return d
