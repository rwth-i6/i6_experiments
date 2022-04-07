from abc import ABC
import dataclasses
import os.path
from sisyphus import tk
from typing import *

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob

from i6_experiments.common.setups.returnn.datastreams.base import Datastream
from i6_experiments.common.setups.returnn.datasets.audio import OggZipDataset


class AdditionalFeatureOptions(ABC):
    pass


@dataclasses.dataclass
class DBMelFilterbankOptions(AdditionalFeatureOptions):
    """
    additional options for the db_mel_filterbank features
    """

    f_min: int = 0
    f_max: int = None
    min_amp: float = 1e-10
    center: bool = True


@dataclasses.dataclass
class MFCCOptions(AdditionalFeatureOptions):
    """
    additional options for the mfcc features
    """

    f_min: int = 0
    f_max: int = None
    n_mels: int = 128


# list of known audio feature type with their respective options type
KNOWN_FEATURES = {
    "mfcc": MFCCOptions,
    "log_mel_filterbank": type(None),
    "log_log_mel_filterbank": type(None),
    "db_mel_filterbank": DBMelFilterbankOptions,
    "linear_spectrogram": type(None),
}


@dataclasses.dataclass
class ReturnnAudioFeatureOptions:
    """
    Commonly used options for RETURNN feature extraction

    :param window_len:
    :param step_len:
    :param num_feature_filters:
    :param with_delta:
    :param features:
    :param additional_feature_options:
    :param sample_rate: audio sample rate, this is not strictly required for RETURNN itself
        but might be needed for certain pipelines
    :param peak_normalization:
    :param preemphasis:
    """

    window_len: float = 0.025
    step_len: float = 0.010
    num_feature_filters: int = None
    with_delta: bool = False
    features: str = "mfcc"
    additional_feature_options: Optional[Union[dict, AdditionalFeatureOptions]] = None
    sample_rate: Optional[int] = None
    peak_normalization: bool = True
    preemphasis: float = None


class AudioFeatureDatastream(Datastream):
    """
    Encapsulates options for audio features used by OggZipDataset via :class:`ExtractAudioFeaturesOptions` in RETURNN
    """

    def __init__(
        self,
        available_for_inference: bool,
        options: ReturnnAudioFeatureOptions,
        **kwargs,
    ):
        """
        :param available_for_inference: define if the DataStream is available during decoding/search. If False,
            it is only available during training.
        :param options: An audio feature options object with the desired feature settings
        :param kwargs: additional options that are passed manually
        """
        super().__init__(available_for_inference)
        self.options = options
        self.additional_options = kwargs.copy()

        if options.features not in KNOWN_FEATURES:
            print("Warning: %s is not a known feature type" % options.features)

        if type(options.additional_feature_options) != KNOWN_FEATURES.get(
            options.features, type(None)
        ):
            print(
                "Warning: possible feature options mismatch, passed %s but expected %s"
                % (
                    str(type(options.additional_feature_options)),
                    str(KNOWN_FEATURES.get(options.features, type(None))),
                )
            )

    def get_feat_dim(self):
        if "num_feature_filters" in self.additional_options:
            return self.additional_options["num_feature_filters"]
        elif self.additional_options["features"] == "raw":
            return 1
        return 40  # some default value

    def as_returnn_extern_data_opts(
        self, available_for_inference: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        :param bool available_for_inference: allows to overwrite the given state if desired. This can be used in case
            the stream is used as output of one model but as input to the next one.
        :return: dictionary for an `extern_data` entry.
        """
        feat_dim = self.options.num_feature_filters
        return {
            **super().as_returnn_extern_data_opts(
                available_for_inference=available_for_inference
            ),
            "shape": (None, feat_dim),
            "dim": feat_dim,
        }

    def as_returnn_audio_opts(self) -> Dict[str, Any]:
        """
        :return: dictionary for `ExtractAudioFeatures` parameters, e.g. as `audio` parameter of the OggZipDataset
        """
        audio_opts_dict = dataclasses.asdict(self.options)
        audio_opts_dict.update(self.additional_options)
        # the additional options itself should not be written as is
        additional_feature_options = audio_opts_dict.pop("additional_feature_options")
        if additional_feature_options is not None:
            audio_opts_dict.update(additional_feature_options)
        return audio_opts_dict


def add_global_statistics_to_audio_feature_datastream(
    audio_datastream: AudioFeatureDatastream,
    zip_datasets: List[tk.Path],
    segment_file: Optional[tk.Path] = None,
    use_scalar_only: bool = False,
    returnn_python_exe: Optional[tk.Path] = None,
    returnn_root: Optional[tk.Path] = None,
    output_path: str = "",
) -> AudioFeatureDatastream:
    """
    Computes the global feature statistics for a given AudioFeatureDatastream over a corpus given as zip-dataset.
    Can either add the statistics per channel (default) or as scalar.

    :param audio_datastream: the audio datastream to which the statistics are added to to which the statistics are added to
    :param zip_datasets: zip dataset which is used for statistics calculation
    :param segment_file: segment file for the dataset
    :param use_scalar_only: use one scalar for mean and variance instead one value per feature channel.
        This is usually done for TTS.
    :param returnn_python_exe:
    :param returnn_root:
    :param output_prefix: sets alias folder for ExtractDatasetStatisticsJob
    :return: audio datastream with added global feature statistics
    :rtype: AudioFeatureDatastream
    """
    extraction_dataset = OggZipDataset(
        path=zip_datasets,
        segment_file=segment_file,
        audio_opts=audio_datastream.as_returnn_audio_opts(),
        target_opts=None,
    )

    extraction_config = ReturnnConfig(
        config={"train": extraction_dataset.as_returnn_opts()}
    )
    extract_dataset_statistics_job = ExtractDatasetMeanStddevJob(
        extraction_config, returnn_python_exe, returnn_root
    )
    extract_dataset_statistics_job.add_alias(
        os.path.join(output_path, "extract_dataset_statistics_job")
    )
    if use_scalar_only:
        audio_datastream.additional_options[
            "norm_mean"
        ] = extract_dataset_statistics_job.out_mean
        audio_datastream.additional_options[
            "norm_std_dev"
        ] = extract_dataset_statistics_job.out_std_dev
    else:
        audio_datastream.additional_options[
            "norm_mean"
        ] = extract_dataset_statistics_job.out_mean_file
        audio_datastream.additional_options[
            "norm_std_dev"
        ] = extract_dataset_statistics_job.out_std_dev_file

    return audio_datastream


def get_default_asr_audio_datastream(
    statistics_ogg_zips: List[tk.Path],
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    output_path: str,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param statistics_ogg_zip: ogg zip file(s) of the training corpus for statistics
    :param returnn_python_exe:
    :param returnn_root:
    :param output_path:
    """
    # default: mfcc-40-dim
    feature_options = ReturnnAudioFeatureOptions(
        window_len=0.025,
        step_len=0.010,
        num_feature_filters=40,
        features="mfcc",
    )
    extract_audio_opts = AudioFeatureDatastream(
        available_for_inference=True, options=feature_options
    )

    audio_datastream = add_global_statistics_to_audio_feature_datastream(
        extract_audio_opts,
        statistics_ogg_zips,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_path=output_path,
    )
    return audio_datastream


def get_default_tts_audio_datastream(
    statistics_ogg_zips: List[tk.Path],
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    output_path: str,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param statistics_ogg_zip: ogg zip file(s) of the training corpus for statistics
    :param returnn_python_exe:
    :param returnn_root:
    :param output_path:
    """
    # default: mfcc-40-dim
    feature_options = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features="db_mel_filterbank",
        peak_normalization=False,
        preemphasis=0.97,
    )
    extract_audio_opts = AudioFeatureDatastream(
        available_for_inference=False, options=feature_options
    )

    audio_datastream = add_global_statistics_to_audio_feature_datastream(
        extract_audio_opts,
        statistics_ogg_zips,
        use_scalar_only=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        output_path=output_path,
    )
    return audio_datastream
