import os
import os.path
from functools import lru_cache
from typing import *
from sisyphus import tk
import asyncio
from abc import ABC
import dataclasses
from enum import Enum
from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob
from i6_core.returnn import ReturnnConfig

from i6_experiments.users.rossenbach.returnn.vocabulary import (
  ReturnnVocabFromBlissLexicon,
)
from i6_experiments.users.rossenbach.setups.returnn_standalone.data.vocabulary import (
  VocabularyDatastream,
)
from i6_experiments.users.hilmes.data.lexicon import get_360_lexicon, get_lexicon
from i6_experiments.common.setups.returnn_common.serialization import (
  DataInitArgs,
  DimInitArgs,
)
from i6_experiments.users.hilmes.data.datasets.audio import (
  OggZipDataset,
)


def get_vocab_360_datastream(alias_path):
  """
  Default VocabularyDatastream for LibriSpeech (uppercase ARPA phoneme symbols)

  :param alias_path:
  :return:
  :rtype: VocabularyDatastream
  """
  uppercase_lexicon = get_360_lexicon()
  returnn_vocab_job = ReturnnVocabFromBlissLexicon(uppercase_lexicon)
  returnn_vocab_job.add_alias(os.path.join(alias_path, "returnn_vocab_from_lexicon"))

  vocab_datastream = VocabularyDatastream(
    available_for_inference=True,
    vocab=returnn_vocab_job.out_vocab,
    vocab_size=returnn_vocab_job.out_vocab_size,
  )

  return vocab_datastream


@lru_cache()
def get_vocab_datastream(alias_path):
  """
  Get the VocabularyDatastream for the ljspeech_sequitur setup

  :param alias_path:
  :return:
  :rtype: VocabularyDatastream
  """
  uppercase_lexicon = get_lexicon()
  returnn_vocab_job = ReturnnVocabFromBlissLexicon(uppercase_lexicon)
  returnn_vocab_job.add_alias(os.path.join(alias_path, "returnn_vocab_from_lexicon"))

  vocab_datastream = VocabularyDatastream(
    available_for_inference=True,
    vocab=returnn_vocab_job.out_vocab,
    vocab_size=returnn_vocab_job.out_vocab_size,
  )

  return vocab_datastream


class Datastream:
  """
  Defines a "Datastream" for a RETURNN setup, meaning a single entry in the "extern_data" dictionary
  of the RETURNN config.
  """

  def __init__(self, available_for_inference: bool):
    """

    :param available_for_inference: "default" value for "available_for_inference"
    """
    self.available_for_inference = available_for_inference

  def as_returnn_extern_data_opts(
    self, available_for_inference: Optional[bool] = None
  ) -> Dict[str, Any]:
    """
    :param available_for_inference: allows to overwrite "available_for_inference" directly
    """
    opts = {
      "available_for_inference": available_for_inference
      if available_for_inference is not None
      else self.available_for_inference,
    }
    return opts

  def as_returnn_common_data_and_dims(
    self, name: str, available_for_inference: Optional[bool] = None, **kwargs
  ):
    """
    :param name (e.g. the datastream key)
    :param available_for_inference:
    :param kwargs:
    """
    d = self.as_returnn_extern_data_opts(
      available_for_inference=available_for_inference
    )
    from returnn_common import nn

    time_dim = nn.SpatialDim("%s_time" % name)
    if isinstance(d["dim"], tk.Variable):
      # for dynamic dims we need asynchron workflows
      asyncio.gather(tk.async_run(d["dim"]))
      dim = d["dim"].get()
    else:
      assert isinstance(d["dim"], int)
      dim = d["dim"]
    if d.get("sparse", False):
      sparse_dim = nn.FeatureDim("%s_indices", dimension=dim)
      data = nn.Data(
        name=name,
        available_for_inference=d["available_for_inference"],
        dim_tags=[nn.batch_dim, time_dim],
        sparse_dim=sparse_dim,
        **kwargs,
      )
      return data, [time_dim, sparse_dim]
    else:
      feature_dim = nn.FeatureDim("%s_feature", dimension=dim)
      data = nn.Data(
        name=name,
        available_for_inference=d["available_for_inference"],
        dim_tags=[nn.batch_dim, time_dim, feature_dim],
        **kwargs,
      )
      return data, [time_dim, feature_dim]

  def as_nnet_constructor_data(
    self, name: str, available_for_inference: Optional[bool] = None, **kwargs
  ):

    d = self.as_returnn_extern_data_opts(
      available_for_inference=available_for_inference
    )
    time_dim = DimInitArgs(
      name="%s_time" % name,
      dim=None,
    )

    dim = d["dim"]

    if d.get("sparse", False):
      sparse_dim = DimInitArgs(name="%s_indices" % name, dim=dim, is_feature=True)
      return DataInitArgs(
        name=name,
        available_for_inference=d["available_for_inference"],
        dim_tags=[time_dim],
        sparse_dim=sparse_dim,
      )
    else:
      feature_dim = DimInitArgs(
        name="%s_feature" % name,
        dim=dim,
        is_feature=True,
      )
      return DataInitArgs(
        name=name,
        available_for_inference=d["available_for_inference"],
        dim_tags=[time_dim, feature_dim],
        sparse_dim=None,
      )


class LabelDatastream(Datastream):
  """
  Defines a datastream for labels represented by indices using the default `Vocabulary` class of RETURNN

  This defines a word-(unit)-based vocabulary
  """

  def __init__(
    self,
    available_for_inference: bool,
    vocab: tk.Path,
    vocab_size: tk.Variable,
    unk_label=None,
  ):
    """

    :param bool available_for_inference:
    :param tk.Path vocab: word vocab file path (pickle)
    :Param tk.Variable|int vocab_size:
    :param str unk_label: unknown label
    """
    super().__init__(available_for_inference)
    self.vocab = vocab
    self.vocab_size = vocab_size
    self.unk_label = unk_label

  def as_returnn_extern_data_opts(
    self, available_for_inference: Optional[bool] = None, **kwargs
  ) -> Dict[str, Any]:
    """
    :param available_for_inference:
    :rtype: dict[str]
    """
    d = {
      **super().as_returnn_extern_data_opts(
        available_for_inference=available_for_inference
      ),
      "shape": (None,),
      "dim": self.vocab_size,
      "sparse": True,
    }
    d.update(kwargs)
    return d

  def as_returnn_targets_opts(self, **kwargs):
    """
    :rtype: dict[str]
    """
    return {"vocab_file": self.vocab, "unknown_label": self.unk_label, **kwargs}


class SpeakerEmbeddingDatastream(Datastream):
  """
  Defines a datastream for speaker embeddings in hdf

  This defines a word-(unit)-based vocabulary
  """

  def __init__(
    self,
    available_for_inference: bool,
    embedding_size: Union[tk.Variable, int],
  ):
    """

    :param bool available_for_inference:
    :Param tk.Variable|int embedding_size:
    """
    super().__init__(available_for_inference)
    self.embedding_size = embedding_size

  def as_returnn_extern_data_opts(
    self, available_for_inference: Optional[bool] = None, **kwargs
  ) -> Dict[str, Any]:
    """
    :param available_for_inference:
    :rtype: dict[str]
    """
    d = {
      **super().as_returnn_extern_data_opts(
        available_for_inference=available_for_inference
      ),
      "shape": (None, self.embedding_size),
      "dim": self.embedding_size,
    }
    d.update(kwargs)
    return d


class DurationDatastream(Datastream):
  """
  Helper class for duration Datastreams
  """

  def as_returnn_extern_data_opts(
    self, available_for_inference: Optional[bool] = None, **kwargs
  ) -> Dict[str, Any]:
    """
    :param available_for_inference:
    :rtype: dict[str]
    """
    d = {
      **super().as_returnn_extern_data_opts(
        available_for_inference=available_for_inference
      ),
      "dim": 1,
      "dtype": "int32",
    }
    d.update(kwargs)
    return d

  def as_nnet_constructor_data(
    self, name: str, available_for_inference: Optional[bool] = None, **kwargs
  ):

    d = self.as_returnn_extern_data_opts(
      available_for_inference=available_for_inference
    )
    time_dim = DimInitArgs(
      name="%s_time" % name,
      dim=None,
    )

    dim = d["dim"]
    feature_dim = DimInitArgs(
      name="%s_feature" % name,
      dim=dim,
      is_feature=True,
    )
    return DataInitArgs(
      name=name,
      available_for_inference=d["available_for_inference"],
      dim_tags=[time_dim, feature_dim],
      sparse_dim=None,
      dtype="int32",
    )


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
  center: bool = True


@dataclasses.dataclass(frozen=True)
class F0Options(AdditionalFeatureOptions):
  """
  additional options for the mfcc features
  """

  fmin: int = 0
  fmax: int = None


# list of known audio feature type with their respective options type
KNOWN_FEATURES = {
  "mfcc": [type(None), MFCCOptions],
  "log_mel_filterbank": [type(None)],
  "log_log_mel_filterbank": [type(None)],
  "db_mel_filterbank": [type(None), DBMelFilterbankOptions],
  "linear_spectrogram": [type(None)],
  "f0": [type(None), F0Options],
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
  F0_FEATURES = "f0"


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
    alias_path: str = "",
  ):
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
      self.additional_options["norm_mean"] = extract_dataset_statistics_job.out_mean
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
