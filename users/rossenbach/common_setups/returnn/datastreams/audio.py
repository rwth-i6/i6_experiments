from abc import ABC
import dataclasses
from enum import Enum
import os.path
from sisyphus import tk
from typing import *

from i6_core.returnn import ReturnnConfig
from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob

from i6_experiments.users.rossenbach.common_setups.returnn.datastreams.base import Datastream
from i6_experiments.users.rossenbach.common_setups.returnn.datasets.audio import OggZipDataset


class AdditionalFeatureOptions(ABC):
    pass


@dataclasses.dataclass(frozen=True)
class DBMelFilterbankOptions(AdditionalFeatureOptions):
    """
    additional options for the db_mel_filterbank features
    """
    fmin: int = 0
    fmax: int = None
    min_amp: float = 1e-10
    center: bool = True


@dataclasses.dataclass(frozen=True)
class MFCCOptions(AdditionalFeatureOptions):
    """
    additional options for the mfcc features
    """
    fmin: int = 0
    fmax: int = None
    n_mels: int = 128


# list of known audio feature type with their respective options type
KNOWN_FEATURES = {
    "mfcc": [type(None), MFCCOptions],
    "log_mel_filterbank": [type(None)],
    "log_log_mel_filterbank": [type(None)],
    "db_mel_filterbank": [type(None), DBMelFilterbankOptions],
    "linear_spectrogram": [type(None)],
}


class FeatureType(Enum):
    """
    Enum helper to have auto-completion for feature types
    """
    MFCC = "mfcc"
    LOG_MEL_FILTERBANK = "log_mel_filterbank"
    LOG_LOG_MEL_FILTERBANK = "log_log_mel_filterbank"
    DB_MEL_FILTERBANK = "db_mel_filterbank"
    LINEAR_SPECTROGRAM = "linear_spectrogram"


@dataclasses.dataclass(frozen=True)
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
    features: Union[str, FeatureType] = "mfcc"
    feature_options: Optional[Union[dict, AdditionalFeatureOptions]] = None
    sample_rate: Optional[int] = None
    peak_normalization: bool = True
    preemphasis: float = None

    def __post_init__(self):
        # convert Enum back to str
        if isinstance(self.features, FeatureType):
            # dataclass is frozen, so directly alter the self.__dict__
            self.__dict__["features"] = self.features.value


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

        if type(options.feature_options) not in KNOWN_FEATURES.get(
            options.features, [type(None)]
        ):
            print(
                "Warning: possible feature options mismatch, passed %s but expected %s"
                % (
                    str(type(options.feature_options)),
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
        return audio_opts_dict


    def add_global_statistics_to_audio_feature_datastream(
            self,
            zip_datasets: List[tk.Path],
            segment_file: Optional[tk.Path] = None,
            use_scalar_only: bool = False,
            returnn_python_exe: Optional[tk.Path] = None,
            returnn_root: Optional[tk.Path] = None,
            alias_path: str = ""):
        """
        Computes the global feature statistics over a corpus given as zip-dataset.
        Can either add the statistics per channel (default) or as scalar.

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
            audio_opts=self.as_returnn_audio_opts(),
            target_opts=None,
        )

        extraction_config = ReturnnConfig(
            config={"train": extraction_dataset.as_returnn_opts()}
        )
        extract_dataset_statistics_job = ExtractDatasetMeanStddevJob(
            extraction_config, returnn_python_exe, returnn_root
        )
        extract_dataset_statistics_job.add_alias(
            os.path.join(alias_path, "extract_dataset_statistics_job")
        )
        if use_scalar_only:
            self.additional_options[
                "norm_mean"
            ] = extract_dataset_statistics_job.out_mean
            self.additional_options[
                "norm_std_dev"
            ] = extract_dataset_statistics_job.out_std_dev
        else:
            self.additional_options[
                "norm_mean"
            ] = extract_dataset_statistics_job.out_mean_file
            self.additional_options[
                "norm_std_dev"
            ] = extract_dataset_statistics_job.out_std_dev_file


def get_default_asr_audio_datastream(
    statistics_ogg_zips: List[tk.Path],
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    alias_path: str,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param statistics_ogg_zip: ogg zip file(s) of the training corpus for statistics
    :param returnn_python_exe:
    :param returnn_root:
    :param alias_path:
    """
    # default: mfcc-40-dim
    feature_options = ReturnnAudioFeatureOptions(
        window_len=0.025,
        step_len=0.010,
        num_feature_filters=40,
        features="mfcc",
    )
    audio_datastream = AudioFeatureDatastream(
        available_for_inference=True, options=feature_options
    )

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        statistics_ogg_zips,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=alias_path,
    )
    return audio_datastream


def get_default_tts_audio_datastream(
    statistics_ogg_zips: List[tk.Path],
    returnn_python_exe: tk.Path,
    returnn_root: tk.Path,
    alias_path: str,
) -> AudioFeatureDatastream:
    """
    Returns the AudioFeatureDatastream using the default feature parameters
    (non-adjustable for now) based on statistics calculated over the provided dataset

    This function serves as an example for ASR Systems, and should be copied and modified in the
    specific experiments if changes to the default parameters are needed

    :param statistics_ogg_zip: ogg zip file(s) of the training corpus for statistics
    :param returnn_python_exe:
    :param returnn_root:
    :param alias_path:
    """
    # default: mfcc-40-dim
    feature_options = ReturnnAudioFeatureOptions(
        window_len=0.050,
        step_len=0.0125,
        num_feature_filters=80,
        features=FeatureType.DB_MEL_FILTERBANK,
        peak_normalization=False,
        preemphasis=0.97,
    )
    audio_datastream = AudioFeatureDatastream(
        available_for_inference=False, options=feature_options
    )

    audio_datastream.add_global_statistics_to_audio_feature_datastream(
        statistics_ogg_zips,
        use_scalar_only=True,
        returnn_python_exe=returnn_python_exe,
        returnn_root=returnn_root,
        alias_path=alias_path,
    )
    return audio_datastream
