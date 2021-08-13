from i6_core.returnn import ReturnnConfig
from i6_experiments.users.rossenbach.setups.returnn_standalone.datasets import OggZipDataset
from i6_private.users.rossenbach.returnn.dataset import ExtractDatasetStatisticsJob


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


def add_feature_dependent_statistics_to_audio_options(
        audio_options,
        ogg_zip_dataset,
        segment_file=None,
        use_scalar_only=False,
        returnn_python_exe=None,
        returnn_root=None,
        output_prefix=""):
    """

    :param AudioFeaturesOpts audio_options:
    :param tk.Path ogg_zip_dataset:
    :param tk.Path segment_file
    :param returnn_python_exe:
    :param returnn_root:
    :param output_prefix:
    :return:
    """

    extraction_dataset = OggZipDataset(
        path=ogg_zip_dataset,
        segment_file=segment_file,
        audio_opts=audio_options.as_returnn_extract_opts(),
        target_opts=None
    )

    extraction_config = ReturnnConfig(config={'train': extraction_dataset.as_returnn_opts()})
    extract_dataset_statistics_job = ExtractDatasetStatisticsJob(extraction_config, returnn_python_exe, returnn_root)
    if use_scalar_only:
        audio_options.options['norm_mean'] = extract_dataset_statistics_job.out_mean
        audio_options.options['norm_std_dev'] = extract_dataset_statistics_job.out_std_dev
    else:
        audio_options.options['norm_mean'] = extract_dataset_statistics_job.out_mean_file
        audio_options.options['norm_std_dev'] = extract_dataset_statistics_job.out_std_dev_file

    return audio_options