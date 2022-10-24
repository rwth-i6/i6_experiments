"""
Librispeech dataset
"""

from __future__ import annotations
from typing import Optional, Union, List, Dict, Any, Tuple
from i6_core.corpus.convert import CorpusToTxtJob
from i6_core.text.label.sentencepiece.train import TrainSentencePieceJob, SentencePieceType
from i6_core.returnn.dataset import ExtractDatasetMeanStddevJob
from i6_core.returnn import ReturnnConfig
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from i6_experiments.common.datasets import librispeech
from .task import Task


librispeech_ogg_zip_dict = librispeech.get_ogg_zip_dict()

# Get Bliss corpus. Same audio format as in ogg_zip, so already there anyway due to how we created the ogg_zip.
bliss_corpus_dict = librispeech.get_bliss_corpus_dict(audio_format="ogg")
bliss_train_corpus = bliss_corpus_dict["train-other-960"]

train_corpus_text = CorpusToTxtJob(bliss_train_corpus, gzip=False).out_txt

# https://github.com/google/sentencepiece/blob/master/doc/options.md
spm_train_job = TrainSentencePieceJob(
  training_text=train_corpus_text,
  vocab_size=2000,
  model_type=SentencePieceType.UNIGRAM,
  additional_options={
    "split_digits": True,
    "unk_id": 2,  # default is 0
    "bos_id": 1,  # default is 1
    "eos_id": 0  # default is 2
  })
spm_2k = spm_train_job.out_model


_Parts = [
    "train-clean-100", "train-clean-360", "train-other-500",
    "dev-clean", "dev-other",
    "test-clean", "test-other"]


# https://github.com/rwth-i6/returnn-experiments/blob/master/2020-librispeech-data-prepare/returnn.config
def _get_dataset(key: str, *, subset=None, train_partition_epoch=None, training: bool = False, targets=None, audio):
  files = []
  parts = [part for part in _Parts if part.startswith(key)]
  assert parts, f"invalid key {key!r}"
  for part in parts:
    files += [librispeech_ogg_zip_dict[part]]
  d = {
    "class": 'OggZipDataset',
    "path": files,
    'use_cache_manager': True,
    "targets": targets,
    "audio": audio,
  }
  if key.startswith("train") and training:
    d["partition_epoch"] = train_partition_epoch
    if key == "train":
      d["epoch_wise_filter"] = {
        (1, 5): {'max_mean_len': 200},
        (6, 10): {'max_mean_len': 500},
      }
    # d["audio"]["random_permute"] = True  # play around. note that this can be slow
    d["seq_ordering"] = "laplace:.1000"
  else:
    d["fixed_random_seed"] = 1
    d["seq_ordering"] = "sorted_reverse"
  if subset:
    d["fixed_random_subset"] = subset  # faster
  return d


# _default_audio_opts_no_stats = dict(features="mfcc", num_feature_filters=40, window_len=0.025, step_len=0.010)
_default_audio_opts_log_mel_fbank_no_stats = dict(
  features="log_mel_filterbank", num_feature_filters=80, window_len=0.025, step_len=0.010)
# _returnn_train_full_no_stats_dict = _get_dataset("train", audio=_default_audio_opts_no_stats)
# _audio_stats_job = ExtractDatasetMeanStddevJob(ReturnnConfig(config={"train": _returnn_train_full_no_stats_dict}))
# default_audio_opts = {
#  **_default_audio_opts_no_stats,
#  "norm_mean": _audio_stats_job.out_mean_file, "norm_std_dev": _audio_stats_job.out_std_dev_file}
default_audio_opts = _default_audio_opts_log_mel_fbank_no_stats

# https://returnn.readthedocs.io/en/latest/api/datasets.util.vocabulary.html#returnn.datasets.util.vocabulary.SentencePieces
default_targets_opts = {
  "class": "SentencePieces",
  "model_file": spm_2k,
  # If your model (e.g. enc-dec) needs EOS, add "add_eos".
}
default_targets_train_opts = default_targets_opts.copy()
default_targets_train_opts.update({
  "enable_sampling": True,  # might be played around with, along with nbest_size, alpha.
})

default_epoch_split = 20

default_dataset_config = {
  "train": _get_dataset(
    "train", training=True, train_partition_epoch=default_epoch_split,
    audio=default_audio_opts, targets=default_targets_train_opts),
  "dev": _get_dataset("dev", subset=3000, audio=default_audio_opts, targets=default_targets_opts),
  "eval_datasets": {
    "devtrain": _get_dataset("train", subset=2000, audio=default_audio_opts, targets=default_targets_opts),
  },
}


class LibrispeechOggZip(DatasetConfig):
  """
  Librispeech dataset in OggZip format.
  """

  def __init__(self, *,
               vocab: Optional[VocabConfig] = None,
               main_key: Optional[str] = None,
               train_epoch_split=default_epoch_split):
    super(LibrispeechOggZip, self).__init__()
    self.vocab = vocab
    self.main_key = main_key
    self.train_epoch_split = train_epoch_split

  # TODO ...


def get_librispeech_task_spm2k() -> Task:
    """
    Librispeech
    """
    # TODO ...
