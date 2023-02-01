"""
Dataset wrapper to allow using the SMS-WSJ dataset in RETURNN.
"""

import functools
import json
import numpy as np
import os.path
import subprocess as sp
from typing import Dict, Tuple, Any, Optional
# noinspection PyUnresolvedReferences
from returnn.datasets.basic import DatasetSeq
# noinspection PyUnresolvedReferences
from returnn.datasets.hdf import HDFDataset
# noinspection PyUnresolvedReferences
from returnn.datasets.map import MapDatasetBase, MapDatasetWrapper
# noinspection PyUnresolvedReferences
from returnn.log import log
# noinspection PyUnresolvedReferences
from returnn.util.basic import OptionalNotImplementedError, NumbersDict
# noinspection PyUnresolvedReferences
from sms_wsj.database import SmsWsj, AudioReader, scenario_map_fn


class SmsWsjBase(MapDatasetBase):
  """
  Base class to wrap the SMS-WSJ dataset. This is not the dataset that is used in the RETURNN config, see
  ``SmsWsjWrapper`` and derived classes for that.
  """

  def __init__(
    self, dataset_name, json_path, pre_batch_transform, buffer=True, zip_cache=None, scenario_map_args=None, **kwargs
  ):
    """
    :param str dataset_name: "train_si284", "cv_dev93" or "test_eval92"
    :param str json_path: path to SMS-WSJ json file
    :param function pre_batch_transform: function which processes raw SMS-WSJ data
    :param bool buffer: if True, use SMS-WSJ dataset prefetching and store sequences in buffer
    :param Optional[str] zip_cache: zip archive with SMS-WSJ data which can be cached, unzipped and used as data dir
    :param Optional[Dict] scenario_map_args: optional kwargs for sms_wsj scenario_map_fn
    """
    super(SmsWsjBase, self).__init__(**kwargs)

    if zip_cache is not None:
      self._cache_zipped_audio(zip_cache, json_path, dataset_name)

    db = SmsWsj(json_path=json_path)
    ds = db.get_dataset(dataset_name)
    ds = ds.map(AudioReader(("original_source", "rir")))

    scenario_map_args = {
      "add_speech_image": False,
      "add_speech_reverberation_early": False,
      "add_speech_reverberation_tail": False,
      "add_noise_image": False,
      **(scenario_map_args or {})}
    ds = ds.map(functools.partial(scenario_map_fn, **scenario_map_args))
    ds = ds.map(functools.partial(pre_batch_transform))

    self._ds = ds
    self._ds_iterator = iter(self._ds)

    self._use_buffer = buffer
    if self._use_buffer:
      self._ds = self._ds.prefetch(4, 8).copy(freeze=True)
    self._buffer = {}  # type Dict[int,[Dict[str,np.array]]]
    self._buffer_size = 40

  def __len__(self) -> int:
    return len(self._ds)

  def __getitem__(self, seq_idx: int) -> Dict[str, np.array]:
    return self._get_seq_by_idx(seq_idx)

  def _get_seq_by_idx(self, seq_idx: int) -> Dict[str, np.array]:
    """
    Returns data for sequence index.
    """
    if self._use_buffer:
      assert seq_idx in self._buffer, f"seq_idx {seq_idx} not in buffer. Available keys are {self._buffer.keys()}"
      return self._buffer[seq_idx]
    else:
      return self._ds[seq_idx]

  def get_seq_tag(self, seq_idx: int) -> str:
    """
    Returns tag for the sequence of the given index, default is 'seq-{seq_idx}'.
    """
    if "seq_tag" in self._get_seq_by_idx(seq_idx):
      return str(self._get_seq_by_idx(seq_idx)["seq_tag"])
    else:
      return "seq-%i" % seq_idx

  def get_seq_len(self, seq_idx: int) -> int:
    """
    Returns length of the sequence of the given index
    """
    if "seq_len" in self._get_seq_by_idx(seq_idx):
      return int(self._get_seq_by_idx(seq_idx)["seq_len"])
    else:
      raise OptionalNotImplementedError

  def get_seq_length_for_keys(self, seq_idx: int) -> NumbersDict:
    """
    Returns sequence length for all data/target keys.
    """
    data = self[seq_idx]
    d = {k: v.size for k, v in data.items()}
    for update_key in ["data", "target_signals"]:
      if update_key in d.keys() and "seq_len" in data:
        d[update_key] = int(data["seq_len"])
    return NumbersDict(d)

  def update_buffer(self, seq_idx: int, pop_seqs: bool = True):
    """
    :param int seq_idx:
    :param bool pop_seqs: if True, pop sequences from buffer that are outside buffer range
    """
    if not self._use_buffer:
      return

    # debugging information
    keys = list(self._buffer.keys()) or [0]
    if not (min(keys) <= seq_idx <= max(keys)):
      print(f"WARNING: seq_idx {seq_idx} outside range of keys: {self._buffer.keys()}")

    # add sequences
    for idx in range(seq_idx, min(seq_idx + self._buffer_size // 2, len(self))):
      if idx not in self._buffer and idx < len(self):
        try:
          self._buffer[idx] = next(self._ds_iterator)
        except StopIteration:
          print(f"StopIteration for seq_idx {seq_idx}")
          print(f"Dataset: {self} with SMS-WSJ {self._ds} of len {len(self)}")
          print(f"Indices in buffer: {self._buffer.keys()}")
          # raise
          print(f"WARNING: Ignoring this, reset iterator and continue")
          self._ds_iterator = iter(self._ds)
          self._buffer[idx] = next(self._ds_iterator)
      if idx == len(self) - 1 and 0 not in self._buffer:
        print(f"Reached end of dataset, reset iterator")
        try:
          next(self._ds_iterator)
        except StopIteration:
          pass
        else:
          print(
            "WARNING: reached final index of dataset, but iterator has more sequences. "
            "Maybe the training was restarted from an epoch > 1?")
        print(f"Current buffer indices: {self._buffer.keys()}")
        self._ds_iterator = iter(self._ds)
        for idx_ in range(self._buffer_size // 2):
          if idx_ not in self._buffer and idx_ < len(self):
            self._buffer[idx_] = next(self._ds_iterator)
        print(f"After adding start of dataset to buffer indices: {self._buffer.keys()}")

    # remove sequences
    if pop_seqs:
      for idx in list(self._buffer):
        if not (seq_idx - self._buffer_size // 2 <= idx <= seq_idx + self._buffer_size // 2):
          if max(self._buffer.keys()) == len(self) - 1 and idx < self._buffer_size // 2:
            # newly added sequences starting from 0
            continue
          self._buffer.pop(idx)

  @staticmethod
  def _cache_zipped_audio(zip_cache: str, json_path: str, dataset_name: str):
    """
    Caches and unzips a given archive with SMS-WSJ data which will then be used as data dir.
    This is done because caching of the single files takes extremely long.
    """
    print(f"Cache and unzip SMS-WSJ data from {zip_cache}")

    # cache and unzip
    try:
      zip_cache_cached = sp.check_output(["cf", zip_cache]).strip().decode("utf8")
      assert zip_cache_cached != zip_cache, "cached and original file have the same path"
      local_unzipped_dir = os.path.dirname(zip_cache_cached)
      sp.check_call(["unzip", "-q", "-n", zip_cache_cached, "-d", local_unzipped_dir])
    except sp.CalledProcessError:
      print(f"Cache manager: Error occurred when caching and unzipping {zip_cache}")
      raise

    # modify json and check if all data is available
    with open(json_path, "r") as f:
      json_dict = json.loads(f.read())
    original_dir = next(iter(json_dict["datasets"][dataset_name].values()))["audio_path"]["original_source"][0]
    while not original_dir.endswith(os.path.basename(local_unzipped_dir)) and len(original_dir) > 1:
      original_dir = os.path.dirname(original_dir)
    for seq in json_dict["datasets"][dataset_name]:
      for audio_key in ["original_source", "rir"]:
        for seq_idx in range(len(json_dict["datasets"][dataset_name][seq]["audio_path"][audio_key])):
          path = json_dict["datasets"][dataset_name][seq]["audio_path"][audio_key][seq_idx]
          path = path.replace(original_dir, local_unzipped_dir)
          json_dict["datasets"][dataset_name][seq]["audio_path"][audio_key][seq_idx] = path
          assert path.startswith(local_unzipped_dir), (
            f"Audio file {path} was expected to start with {local_unzipped_dir}")
          assert os.path.exists(path), f"Audio file {path} does not exist"

    json_path = os.path.join(local_unzipped_dir, "sms_wsj.json")
    with open(json_path, "w", encoding="utf-8") as f:
      json.dump(json_dict, f, ensure_ascii=False, indent=4)

    print(f"Finished preparation of zip cache data, use json in {json_path}")


class SmsWsjBaseWithRasrClasses(SmsWsjBase):
  """
  Base class to wrap the SMS-WSJ dataset and combine it with RASR alignments in an hdf dataset.
  """

  def __init__(
      self, rasr_classes_hdf=None, rasr_corpus=None, rasr_segment_prefix="", rasr_segment_postfix="", **kwargs
  ):
    """
    :param Optional[str] rasr_classes_hdf: hdf file with dumped RASR class labels
    :param Optional[str] rasr_corpus: RASR corpus file for reading segment start and end times for padding
    :param str rasr_segment_prefix: prefix to map SMS-WSJ segment name to RASR segment name
    :param str rasr_segment_postfix: postfix to map SMS-WSJ segment name to RASR segment name
    :param kwargs:
    """
    super(SmsWsjBaseWithRasrClasses, self).__init__(**kwargs)

    # self.data_types = {"target_signals": {"dim": 2, "shape": (None, 2)}}
    self._rasr_classes_hdf = None
    if rasr_classes_hdf is not None:
      self._rasr_classes_hdf = HDFDataset([rasr_classes_hdf], use_cache_manager=True)
      # self.data_types["target_rasr"] = {"sparse": True, "dim": 9001, "shape": (None, 2)}
    self._rasr_segment_start_end = {}  # type: Dict[str, Tuple[float, float]]
    if rasr_corpus is not None:
      from i6_core.lib.corpus import Corpus
      corpus = Corpus()
      corpus.load(rasr_corpus)
      for seg in corpus.segments():
        self._rasr_segment_start_end[seg.fullname()] = (seg.start, seg.end)
    self.rasr_segment_prefix = rasr_segment_prefix
    self.rasr_segment_postfix = rasr_segment_postfix

  def __getitem__(self, seq_idx: int) -> Dict[str, np.array]:
    d = self._get_seq_by_idx(seq_idx)
    if self._rasr_classes_hdf is not None:
      rasr_seq_tags = [
        f"{self.rasr_segment_prefix}{d['seq_tag']}_{speaker}{self.rasr_segment_postfix}"
        for speaker in range(d["target_signals"].shape[1])]
      rasr_targets = []
      for idx, rasr_seq_tag in enumerate(rasr_seq_tags):
        rasr_targets.append(self._rasr_classes_hdf.get_data_by_seq_tag(rasr_seq_tag, "classes"))
      start_end_times = [self._rasr_segment_start_end.get(rasr_seq_tag, (0, 0)) for rasr_seq_tag in rasr_seq_tags]
      padded_len_sec = max(start_end[1] for start_end in start_end_times)
      padded_len_frames = max(rasr_target.shape[0] for rasr_target in rasr_targets)
      for speaker_idx in range(len(rasr_targets)):
        pad_start = 0
        if padded_len_sec > 0:
          pad_start = round(start_end_times[speaker_idx][0] / padded_len_sec * padded_len_frames)
        pad_end = padded_len_frames - rasr_targets[speaker_idx].shape[0] - pad_start
        if pad_end < 0:
          pad_start += pad_end
          assert pad_start >= 0
          pad_end = 0
        rasr_targets[speaker_idx] = np.concatenate([
          9000 * np.ones(pad_start), rasr_targets[speaker_idx], 9000 * np.ones(pad_end)])
      d["target_rasr"] = np.stack(rasr_targets).T
      d["target_rasr_len"] = np.array(padded_len_frames)
    return d

  def get_seq_length_for_keys(self, seq_idx: int) -> NumbersDict:
    """
    Returns sequence length for all data/target keys.
    """
    d = super(SmsWsjBaseWithRasrClasses, self).get_seq_length_for_keys(seq_idx)
    data = self[seq_idx]
    d["target_rasr"] = int(data["target_rasr_len"])
    return NumbersDict(d)


class SmsWsjWrapper(MapDatasetWrapper):
  """
  Base class for datasets that can be used in RETURNN config.
  """

  def __init__(
      self, sms_wsj_base, **kwargs
  ):
    """
    :param Optional[SmsWsjBase] sms_wsj_base: SMS-WSJ base class to allow inherited classes to modify this
    """
    if "seq_ordering" not in kwargs:
      print("Warning: no shuffling is enabled by default", file=log.v)
    super(SmsWsjWrapper, self).__init__(sms_wsj_base, **kwargs)
    # self.num_outputs = ...  # needs to be set in derived classes

    def _get_seq_length(seq_idx: int) -> NumbersDict:
      """
      Returns sequence length for all data/target keys.
      """
      corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)
      return sms_wsj_base.get_seq_length_for_keys(corpus_seq_idx)

    self.get_seq_length = _get_seq_length

  @staticmethod
  def _pre_batch_transform(inputs: Dict[str, Any]) -> Dict[str, np.array]:
    """
    Used to process raw SMS-WSJ data
    :param  inputs: input as coming from SMS-WSJ
    """
    return_dict = {
      "seq_tag": np.array(inputs["example_id"], dtype=object),
      "source_id": np.array(inputs["source_id"], dtype=object),
      "seq_len": np.array(inputs["num_samples"]["observation"]),
    }
    return return_dict

  def _collect_single_seq(self, seq_idx: int) -> DatasetSeq:
    """
    :param seq_idx: sorted seq idx
    """
    corpus_seq_idx = self.get_corpus_seq_idx(seq_idx)
    self._dataset.update_buffer(corpus_seq_idx)
    data = self._dataset[corpus_seq_idx]
    assert "seq_tag" in data
    return DatasetSeq(seq_idx, features=data, seq_tag=data["seq_tag"])

  def init_seq_order(self, epoch: Optional[int] = None, **kwargs) -> bool:
    """
    Override this in order to update the buffer. get_seq_length is often called before _collect_single_seq, therefore
    the buffer does not contain the initial indices when continuing the training from an epoch > 0.
    """
    out = super(SmsWsjWrapper, self).init_seq_order(epoch=epoch, **kwargs)
    buffer_index = ((epoch or 1) - 1) * self.num_seqs % len(self._dataset)
    self._dataset.update_buffer(buffer_index, pop_seqs=False)
    return out


class SmsWsjMixtureEarlyDataset(SmsWsjWrapper):
  """
  Dataset with audio mixture and early signals as target.
  """

  def __init__(
      self, dataset_name, json_path, num_outputs=None, zip_cache=None, sms_wsj_base=None, **kwargs
  ):
    """
    :param str dataset_name: "train_si284", "cv_dev93" or "test_eval92"
    :param str json_path: path to SMS-WSJ json file
    :param Optional[Dict[str, List[int]]] num_outputs: num_outputs for RETURNN dataset
    :param Optional[str] zip_cache: zip archive with SMS-WSJ data which can be cached, unzipped and used as data dir
    :param Optional[SmsWsjBase] sms_wsj_base: SMS-WSJ base class to allow inherited classes to modify this
    """
    if sms_wsj_base is None:
      sms_wsj_base = SmsWsjBase(
        dataset_name=dataset_name,
        json_path=json_path,
        pre_batch_transform=self._pre_batch_transform,
        scenario_map_args={"add_speech_reverberation_early": True},
        zip_cache=zip_cache)
    super(SmsWsjMixtureEarlyDataset, self).__init__(sms_wsj_base, **kwargs)
    # typically data is raw waveform so 1-D and dense, target signals are 2-D (one for each speaker) and dense
    self.num_outputs = num_outputs or {"data": [1, 2], "target_signals": [2, 2]}

  @staticmethod
  def _pre_batch_transform(inputs: Dict[str, Any]) -> Dict[str, np.array]:
    """
    Used to process raw SMS-WSJ data
    :param inputs: input as coming from SMS-WSJ
    """
    return_dict = SmsWsjWrapper._pre_batch_transform(inputs)
    return_dict.update({
      "data": inputs["audio_data"]["observation"][:1, :].T,  # take first of 6 channels: (T, 1)
      "target_signals": inputs["audio_data"]["speech_reverberation_early"][:, 0, :].T,  # first of 6 channels: (T, S)
    })
    return return_dict


class SmsWsjMixtureEarlyAlignmentDataset(SmsWsjMixtureEarlyDataset):
  """
  Dataset with audio mixture, target early signals and target RASR alignments.
  """

  def __init__(
      self, dataset_name, json_path, num_outputs=None, rasr_num_outputs=None, rasr_segment_prefix="",
      rasr_segment_postfix="", rasr_classes_hdf=None, rasr_corpus=None, zip_cache=None, **kwargs
  ):
    """
    :param str dataset_name: "train_si284", "cv_dev93" or "test_eval92"
    :param str json_path: path to SMS-WSJ json file
    :param Optional[Dict[str, List[int]]] num_outputs: num_outputs for RETURNN dataset
    :param Optional[int] rasr_num_outputs: number of output labels for RASR alignment, e.g. 9001 for that CART size
    :param str rasr_segment_prefix: prefix to map SMS-WSJ segment name to RASR segment name
    :param str rasr_segment_postfix: postfix to map SMS-WSJ segment name to RASR segment name
    :param str rasr_classes_hdf: hdf file with dumped RASR class labels
    :param str rasr_corpus: RASR corpus file for reading segment start and end times for padding
    :param Optional[str] zip_cache: zip archive with SMS-WSJ data which can be cached, unzipped and used as data dir
    """
    sms_wsj_base = SmsWsjBaseWithRasrClasses(
      dataset_name=dataset_name,
      json_path=json_path,
      pre_batch_transform=self._pre_batch_transform,
      scenario_map_args={"add_speech_reverberation_early": True},
      rasr_classes_hdf=rasr_classes_hdf,
      rasr_corpus=rasr_corpus,
      rasr_segment_prefix=rasr_segment_prefix,
      rasr_segment_postfix=rasr_segment_postfix,
      zip_cache=zip_cache)
    super(SmsWsjMixtureEarlyAlignmentDataset, self).__init__(
      dataset_name, json_path, num_outputs=num_outputs, zip_cache=zip_cache, sms_wsj_base=sms_wsj_base, **kwargs)
    if num_outputs is not None:
      self.num_outputs = num_outputs
    else:
      assert rasr_num_outputs is not None, "either num_outputs or rasr_num_outputs has to be given"
      self.num_outputs["target_rasr"] = [rasr_num_outputs, 1]  # target alignments are sparse with the given dim
