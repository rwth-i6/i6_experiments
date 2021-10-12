"""
Helper classes around RETURNN datasets
"""
from sisyphus import tk
from typing import *


class GenericDataset:
    """
    Basic interface for all datasets
    """

    def as_returnn_opts(self):
        """
        return data dict for SprintDataset, OggZipDataset, etc
        :return: dict[str]
        """
        raise NotImplementedError


class ControlDataset(GenericDataset):
    """
    A template for all Datasets that allow for data control (sequence ordering, partitioning, subsets)
    """

    def __init__(
            self,
            subset: Optional[int] = None,
            partition_epoch: Optional[int] = None,
            segment_file: Optional[tk.Path] = None,
            seq_ordering: Optional[str] = None):
        self.subset = subset
        self.partition_epoch = partition_epoch or 1
        self.segment_file = segment_file
        self.seq_ordering = seq_ordering

    def as_returnn_opts(self):
        d = {
            "partition_epoch": self.partition_epoch,
        }
        if self.seq_ordering:
            d['seq_ordering'] = self.seq_ordering
        if self.segment_file:
            d['segment_file'] = self.segment_file
        if self.subset:
            d['fixed_random_subset'] = self.subset  # faster
        return d


class HDFDataset(ControlDataset):

    def __init__(
            self,
            files: Union[List[tk.Path], tk.Path],
            subset: Optional[int] = None,
            partition_epoch: Optional[int] = None,
            segment_file: Optional[tk.Path] = None,
            seq_ordering: Optional[str] = None,
            other_opts: Optional[Dict[str, Any]] = None):
        super().__init__(subset, partition_epoch, segment_file, seq_ordering)
        self.files = files
        self.segment_file = segment_file
        self.seq_ordering = seq_ordering
        if other_opts is None:
            other_opts = {}
        else:
            other_opts = other_opts.copy()
        self.other_opts = other_opts


    def as_returnn_opts(self):
        d = {
            'class': "HDFDataset",
            'files': self.files if isinstance(self.files, list) else [self.files],
            'use_cache_manager': True,
        }
        d.update(super().as_returnn_opts())
        d.update(self.other_opts)
        return d


class OggZipDataset(ControlDataset):
    """
    Represents :class:`OggZipDataset` in RETURNN
    `BlissToOggZipJob` job is used to convert some bliss xml corpus to ogg zip files
    """

    def __init__(self,
                 path: Union[List[tk.Path], tk.Path],
                 audio_opts: Optional[Dict[str, Any]] = None,
                 target_opts: Optional[Dict[str, Any]] = None,
                 subset: Optional[int] = None,
                 partition_epoch: Optional[int] = None,
                 segment_file: Optional[tk.Path] = None,
                 seq_ordering: Optional[str] = None,
                 other_opts: Optional[Dict[str, Any]] = None):
        """
        :param path: ogg zip files path
        :param audio_opts: used to for feature extraction
        :param target_opts: used to create target labels
        :param partition_epoch: set explicitly here otherwise it is set to 1 later
        :param other_opts: other opts for OggZipDataset RETURNN class
        """
        super().__init__(subset, partition_epoch, segment_file, seq_ordering)
        self.path = path
        self.audio_opts = audio_opts
        self.target_opts = target_opts
        if other_opts is None:
            other_opts = {}
        else:
            other_opts = other_opts.copy()
        assert 'audio' not in other_opts
        assert 'targets' not in other_opts
        assert 'partition_epoch' not in other_opts
        self.other_opts = other_opts

    def as_returnn_opts(self):
        d = {
            'class': 'OggZipDataset',
            'path': self.path[0] if isinstance(self.path, list) and len(self.path) == 1 else self.path,
            'use_cache_manager': True,
            'audio': self.audio_opts,
            'targets': self.target_opts,
        }
        d.update(super().as_returnn_opts())
        d.update(self.other_opts)
        return d


class MetaDataset(GenericDataset):
    """
    Represents :class:`MetaDataset` in RETURNN

    Only allows the MetaDataset to be used with an explicit control dataset.
    """

    def __init__(self,
                 data_map: Dict[str, Tuple[str, str]],
                 datasets: Dict[str, Union[Dict, GenericDataset]],
                 seq_order_control_dataset: str,
                 other_opts: Optional[Dict[str, Any]] = None):
        """
        :param data_map:
        :param datasets:
        :param seq_order_control_dataset:
        :param dict other_opts:
        """
        self.data_map = data_map
        self.datasets = {k: v if isinstance(v, dict) else v.as_returnn_opts() for k, v in datasets.items()}
        assert seq_order_control_dataset in datasets
        self.seq_order_control_dataset = seq_order_control_dataset
        if other_opts is None:
            other_opts = {}
        self.other_opts = other_opts

    def as_returnn_opts(self):
        d = {
            'class': 'MetaDataset',
            'data_map': self.data_map,
            'datasets': self.datasets,
            'seq_order_control_dataset': self.seq_order_control_dataset
        }
        d.update(self.other_opts)
        return d

