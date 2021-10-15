class GenericDataset:

    def as_returnn_opts(self):
        """
        return data dict for SprintDataset, OggZipDataset, etc
        :return: dict[str]
        """
        raise NotImplementedError


class OggZipDataset(GenericDataset):
    """
    Represents :class:`OggZipDataset` in RETURNN
    `BlissToOggZipJob` job is used to convert some bliss xml corpus to ogg zip files
    """

    def __init__(self, path, audio_opts=None, target_opts=None, subset=None, partition_epoch=None, segment_file=None,
                 seq_ordering=None,
                 other_opts=None):
        """
        :param List[Path|str]|Path|str path: ogg zip files path
        :param dict[str]|None audio_opts: used to for feature extraction
        :param dict[str]|None target_opts: used to create target labels
        :param int|None subset: only use a subset of random N sequences
        :param int|None partition_epoch: set explicitly here otherwise it is set to 1 later
        :param Path|None segment_file: path to a line based segment file
        :param str|None seq_ordering: sequence ordering mode definition
        :param dict[str]|None other_opts: other opts for OggZipDataset RETURNN class
        """
        self.path = path
        self.audio_opts = audio_opts
        self.target_opts = target_opts
        self.subset = subset
        self.partition_epoch = partition_epoch or 1
        self.segment_file = segment_file
        self.seq_ordering = seq_ordering
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
            'path': self.path,
            'use_cache_manager': True,
            'audio': self.audio_opts,
            'targets': self.target_opts,
            'partition_epoch': self.partition_epoch
        }
        if self.seq_ordering:
            d['seq_ordering'] = self.seq_ordering
        if self.segment_file:
            d['segment_file'] = self.segment_file
        if self.subset:
            d['fixed_random_subset'] = self.subset  # faster
        d.update(self.other_opts)
        return d


class MetaDataset(GenericDataset):
    """
    Represents `:class:MetaDataset` in RETURNN
    """

    def __init__(self, data_map, datasets, seq_order_control_dataset, other_opts=None):
        """
        :param dict[str, tuple(str, str)] data_map: datastream -> (dataset_name, datastream),
            mappings of the datastream of specific datasets to a global datastream identifier
        :param dict[str, Union[dict, GenericDataset]] datasets:
        :param str seq_order_control_dataset:
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

