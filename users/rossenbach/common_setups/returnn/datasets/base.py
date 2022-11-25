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

    def __init__(self, *, additional_options: Optional[Dict]):
        """
        :param additional_options: any options not explicitly covered by the helper classes or for debugging purposes
        """
        self.additional_options = additional_options

    def as_returnn_opts(self) -> Dict[str, Any]:
        """
        :return: data dict for SprintDataset, OggZipDataset, etc
        """
        return self.additional_options or {}


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
        random_subset: Optional[int] = None,
        # super parameters
        additional_options: Optional[Dict] = None,
    ):
        """
        :param partition_epoch: partition the data into N parts
        :param seq_list_filter_file: text file (gzip/plain) or pkl containg list of sequence tags to use
        :param seq_ordering: see `https://returnn.readthedocs.io/en/latest/dataset_reference/index.html`_.
        :param random_subset: take a random subset of the data, this is typically used for "dev-train", a part
            of the training data which is used to see training scores without data augmentation
        :param additional_options: custom options directly passed to the dataset
        """
        super().__init__(additional_options=additional_options)
        self.partition_epoch = partition_epoch or 1
        self.seq_list_filter_file = seq_list_filter_file
        self.seq_ordering = seq_ordering
        self.random_subset = random_subset

    def as_returnn_opts(self) -> Dict[str, Any]:
        d = {
            "partition_epoch": self.partition_epoch,
        }
        if self.seq_ordering:
            d["seq_ordering"] = self.seq_ordering
        if self.seq_list_filter_file:
            d["seq_list_filter_file"] = self.seq_list_filter_file
        if self.random_subset:
            d['fixed_random_subset'] = self.random_subset
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


    Example code for the MetaDataset:

        meta_dataset = datasets.MetaDataset(
            data_map={'audio_features': ('audio', 'data'),
                      'phon_labels': ('audio', 'classes'),
                      'speaker_labels': ('speaker', 'data'),
                      },
            datasets={
                'audio': audio_dataset.as_returnn_opts(),  # OggZipDataset
                'speaker': speaker_dataset.as_returnn_opts()  # HDFDataset
            },
            seq_order_control_dataset="audio",
        )
    """

    def __init__(
        self,
        *,
        data_map: Dict[str, Tuple[str, str]],
        datasets: Dict[str, Union[Dict, GenericDataset]],
        seq_order_control_dataset: str,
        additional_options: Optional[Dict[str, Any]] = None,
    ):
        """
        :param data_map: datastream mapping with "<extern_data_name>": ("<dataset_name>", "<dataset_output>")
            Example "audio_data": ("audio_hdf_dataset", "data")
        :param datasets: dictionary with "<dataset_name>": <dataset_option_dict>
        :param seq_order_control_dataset: which dataset identified by name to use for sequence ordering
        :param dict additional_options: additional options to be passed to the meta dataset
        """
        super().__init__(additional_options=additional_options)
        self.data_map = data_map
        self.datasets = {
            k: v if isinstance(v, dict) else v.as_returnn_opts()
            for k, v in datasets.items()
        }
        assert seq_order_control_dataset in datasets
        self.seq_order_control_dataset = seq_order_control_dataset

    def as_returnn_opts(self) -> Dict[str, Any]:
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
