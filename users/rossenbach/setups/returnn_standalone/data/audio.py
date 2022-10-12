import copy
import os.path
from functools import lru_cache
from sisyphus import tk
from typing import *

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob

from i6_experiments.users.rossenbach.setups import returnn_standalone
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.datasets import OggZipDataset
from i6_experiments.users.rossenbach.returnn.dataset import ExtractDatasetStatisticsJob

from .common import Datastream


class AudioFeatureDatastream(Datastream):
    """
    Encapsulates options for audio features used by OggZipDataset via :class:`ExtractAudioFeaturesOptions` in RETURNN
    """

    _default_options = dict(window_len=0.025, step_len=0.010, features="mfcc")

    def __init__(self, available_for_inference: bool, sample_rate: Optional[int] = None, **kwargs):
        """
        :param available_for_inference: define if the DataStream is available during decoding/search. If False,
            it is only available during training.
        :param sample_rate: audio sample rate, this is not strictly required but might be needed for certain pipelines
        :param kwargs:
        """
        super().__init__(available_for_inference)
        self.options = kwargs.copy()
        self.sample_rate = sample_rate
        for k, v in self._default_options.items():
            self.options.setdefault(k, v)

    def get_feat_dim(self):
        if "num_feature_filters" in self.options:
            return self.options["num_feature_filters"]
        elif self.options["features"] == "raw":
            return 1
        return 40  # some default value

    def as_returnn_extern_data_opts(self, available_for_inference: Optional[bool] = None) -> Dict[str, Any]:
        feat_dim = self.get_feat_dim()
        return {
            **super().as_returnn_extern_data_opts(available_for_inference=available_for_inference),
            "shape": (None, feat_dim),
            "dim": feat_dim,
        }

    def as_returnn_audio_opts(self) -> Dict[str, Any]:
        return copy.deepcopy(self.options)


def add_global_statistics_to_audio_features(
    audio_datastream: AudioFeatureDatastream,
    zip_dataset: List[tk.Path],
    segment_file: Optional[tk.Path] = None,
    use_scalar_only: bool = False,
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    alias_path: str = "",
) -> AudioFeatureDatastream:
    """
    Computes the global statistics for a given AudioFeatureDatastream over a corpus given as zip-dataset

    :param audio_datastream: the audio datastream to which the statistics are added to to which the statistics are added to
    :param zip_dataset: zip dataset which is used for statistics calculation
    :param segment_file: segment file for the dataset
    :param returnn_python_exe:
    :param returnn_root:
    :param alias_path: sets alias folder for ExtractDatasetStatisticsJob
    :return: audio datastream with added global feature statistics
    :rtype: AudioFeatureDatastream
    """
    extraction_dataset = OggZipDataset(
        path=zip_dataset,
        segment_file=segment_file,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=None,
    )

    extraction_config = ReturnnConfig(config={"train": extraction_dataset.as_returnn_opts()})
    extract_dataset_statistics_job = ExtractDatasetStatisticsJob(extraction_config, returnn_python_exe, returnn_root)
    extract_dataset_statistics_job.add_alias(os.path.join(alias_path, "extract_statistics_job"))
    if use_scalar_only:
        audio_datastream.options["norm_mean"] = extract_dataset_statistics_job.out_mean
        audio_datastream.options["norm_std_dev"] = extract_dataset_statistics_job.out_std_dev
    else:
        audio_datastream.options["norm_mean"] = extract_dataset_statistics_job.out_mean_file
        audio_datastream.options["norm_std_dev"] = extract_dataset_statistics_job.out_std_dev_file

    return audio_datastream


def add_global_statistics_to_audio_datastream(
    audio_datastream,
    zip_dataset,
    segment_file=None,
    use_scalar_only=False,
    returnn_python_exe=None,
    returnn_root=None,
    output_prefix="",
):
    """
    updated compared to  add_global_statistics_to_audio_features

    :param AudioFeaturesOpts audio_options:
    :param tk.Path|List[tk.Path] zip_dataset:
    :param tk.Path segment_file
    :param returnn_python_exe:
    :param returnn_root:
    :param output_prefix:
    :return:
    """

    extraction_dataset = OggZipDataset(
        path=zip_dataset,
        segment_file=segment_file,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=None,
    )

    extraction_config = ReturnnConfig(config={"train": extraction_dataset.as_returnn_opts()})
    extract_dataset_statistics_job = ExtractDatasetMeanStddevJob(extraction_config, returnn_python_exe, returnn_root)
    extract_dataset_statistics_job.add_alias(os.path.join(output_prefix, "extract_dataset_statistics_job"))
    if use_scalar_only:
        audio_datastream.options["norm_mean"] = extract_dataset_statistics_job.out_mean
        audio_datastream.options["norm_std_dev"] = extract_dataset_statistics_job.out_std_dev
    else:
        audio_datastream.options["norm_mean"] = extract_dataset_statistics_job.out_mean_file
        audio_datastream.options["norm_std_dev"] = extract_dataset_statistics_job.out_std_dev_file

    return audio_datastream


@lru_cache()
def get_default_asr_audio_datastream(statistics_ogg_zip, returnn_python_exe, returnn_root, output_path):
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param Path statistics_ogg_zip: ogg zip file of the training corpus for statistics
    :param Path returnn_python_exe:
    :param Path returnn_root:
    :param str output_path:
    :return: returnn_standalone.data.audio.AudioFeatureDatastream
    """
    # default: mfcc-40-dim
    extract_audio_opts = returnn_standalone.data.audio.AudioFeatureDatastream(
        available_for_inference=True, window_len=0.025, step_len=0.010, num_feature_filters=40, features="mfcc"
    )

    audio_datastream = returnn_standalone.data.audio.add_global_statistics_to_audio_features(
        extract_audio_opts,
        statistics_ogg_zip,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=output_path,
    )
    return audio_datastream
