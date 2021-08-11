

class AudioFeaturesOpts:
    """
    Encapsulates options for audio features used by OggZipDataset via :class:ExtractAudioFeaturesOptions in RETURNN
    """

    _default_options = dict(window_len=0.025, step_len=0.010, features='mfcc')

    def __init__(self, available_for_inference, **kwargs):
        self.available_for_inference = available_for_inference
        self.options = kwargs.copy()
        for k, v in self._default_options.items():
            self.options.setdefault(k, v)

    def get_feat_dim(self):
        if 'num_feature_filters' in self.options:
            return self.options['num_feature_filters']
        elif self.options['features'] == 'raw':
            return 1
        return 40  # some default value

    def as_returnn_data_opts(self):
        """

        :return:
        :rtype: dict[str, Any]
        """
        feat_dim = self.get_feat_dim()
        return {'shape': (None, feat_dim), 'dim': feat_dim, 'available_for_inference': self.available_for_inference}

    def as_returnn_extract_opts(self):
        return self.options


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

    def __init__(self, path, audio_opts=None, target_opts=None, subset=None, epoch_split=None, segment_file=None,
                 other_opts=None):
        """
        :param List[Path|str]|Path|str path: ogg zip files path
        :param dict[str]|None audio_opts: used to for feature extraction
        :param dict[str]|None target_opts: used to create target labels
        :param int|None epoch_split: set explicitly here otherwise it is set to 1 later
        :param dict[str]|None other_opts: other opts for OggZipDataset RETURNN class
        """
        self.path = path
        self.audio_opts = audio_opts
        self.target_opts = target_opts
        self.subset = subset
        self.epoch_split = epoch_split or 1
        self.segment_file = segment_file
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
            'partition_epoch': self.epoch_split
        }
        if self.segment_file:
            d['segment_file'] = self.segment_file
        if self.subset:
            d['fixed_random_subset'] = self.subset  # faster
        d.update(self.other_opts)
        return d

