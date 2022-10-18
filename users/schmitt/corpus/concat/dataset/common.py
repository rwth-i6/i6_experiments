

from sisyphus import *
from sisyphus.tools import sis_hash
from sisyphus.hash import sis_hash_helper
from recipe.i6_experiments.users.schmitt.experiments.config.concat_seqs.utils import generic_open
import os
import sys
import typing


class CreateBpeJob(Job):
  Existing = {}  # type: typing.Dict[bytes,BpeCodesVocab]  # hash -> codes and vocab

  def __init__(self, txt_file, bpe_size):
    """
    :param Path txt_file:
    :param int bpe_size:
    """
    self.txt_file = txt_file
    self.bpe_size = bpe_size
    key = sis_hash_helper((self.txt_file, self.bpe_size))
    if key in self.Existing:
      self._have_existing = True
      self.output = self.Existing[key]
      self.output_bpe_codes = self.output.codes
      self.output_bpe_vocab = self.output.vocab
    else:
      self._have_existing = False
      self.output_bpe_codes = self.output_path("bpe.codes")
      self.output_bpe_vocab = self.output_path("bpe.vocab")
      self.output = BpeCodesVocab(codes=self.output_bpe_codes, vocab=self.output_bpe_vocab, unk="<unk>")

  @classmethod
  def register_existing(cls, txt_file, bpe_size, output):
    """
    :param Path txt_file:
    :param int bpe_size:
    :param BpeCodesVocab output:
    """
    if output.vocab.available() and output.codes.available():
      cls.Existing[sis_hash_helper((txt_file, bpe_size))] = output

  def run(self):
    cmd_cat_txt = "%s %s" % ("zcat" if self.txt_file.path.endswith(".gz") else "cat", self.txt_file.get_path())
    self.sh(
      "%s | %s/subword-nmt/learn_bpe.py --output %s --symbols %s" % (
        cmd_cat_txt, tk.gs.BASE_DIR, self.output_bpe_codes.get_path(), self.bpe_size))
    self.sh(
      '%s/subword-nmt/create-py-vocab.py --txt %s --bpe %s --unk "<unk>" --out %s' % (
        tk.gs.BASE_DIR, self.txt_file, self.output_bpe_codes, self.output_bpe_vocab.get_path()))

  def tasks(self):
    if self._have_existing:
      return  # nothing to do
    yield Task('run', rqmt={'mem': 1, 'time': 5}, mini_task=True)


def txt_to_bpe(txt_file, bpe_size):
  """
  :param Path txt_file:
  :param int bpe_size:
  :rtype: BpeCodesVocab
  """
  return CreateBpeJob(txt_file=txt_file, bpe_size=bpe_size).output


class Targets:
  """
  This corresponds to the ``targets`` (or ``bpe``) options in some RETURNN datasets,
  e.g. :class:`returnn.OggZipDataset`.
  """

  def register_output(self, prefix):
    """
    :param str prefix:
    """
    pass  # optional

  def as_returnn_data_opts(self):
    """
    :return: opts for :class:`returnn.Data` in :class:`returnn.ExternData`.
      Can return None if not yet available (we don't know about the dim in that case).
      Used for the RETURNN config as ``extern_data``.
    :rtype: dict[str]|None
    """
    raise NotImplementedError

  def as_returnn_dataset_targets_opts(self):
    """
    :return: opts for :class:`returnn.Vocabulary.create_vocab`, e.g. in :class:`returnn.OggZipDataset`
    :rtype: dict[str]
    """
    raise NotImplementedError


class VocabBasedTargets(Targets):
  def __init__(self, vocab, unk=None):
    """
    :param Path|str vocab:
    :param str|None unk:
    """
    self.vocab = vocab if isinstance(vocab, Path) else Path(vocab)
    self.unk = unk

  def as_returnn_data_opts(self):
    """
    :rtype: dict[str]|None
    """
    if self.vocab.available():
      from returnn.GeneratingDataset import Vocabulary
      # This would cache the vocab by path, thus it's fast.
      vocab = Vocabulary(vocab_file=self.vocab.get_path(), unknown_label=self.unk)
      dim = vocab.num_labels
      return {
        "shape": (None,), "dim": dim, "sparse": True,
        # Fallback to simpler default, which just loads the vocab itself, but not the BPE codes (faster).
        "vocab": {"vocab_file": self.vocab.get_path(), "unknown_label": self.unk}}
    return None

  def as_returnn_dataset_targets_opts(self):
    """
    :rtype: dict[str]
    """
    return {
      "vocab_file": self.vocab,
      "unknown_label": self.unk}


class BpeCodesVocab(VocabBasedTargets):
  def __init__(self, codes, vocab, unk=None):
    """
    :param Path|str codes:
    :param Path|str vocab:
    :param str|None unk:
    """
    super(BpeCodesVocab, self).__init__(vocab=vocab, unk=unk)
    self.codes = codes if isinstance(codes, Path) else Path(codes)

  def register_output(self, prefix):
    """
    :param str prefix:
    """
    tk.register_output("%s.codes" % prefix, self.codes)
    tk.register_output("%s.vocab" % prefix, self.vocab)

  def as_returnn_dataset_targets_opts(self):
    """
    :rtype: dict[str]
    """
    return {
      "class": "BytePairEncoding",
      "vocab_file": self.vocab,
      "bpe_file": self.codes,
      "unknown_label": self.unk}


class ExtractAudioFeaturesOptions:
  """
  This corresponds to the options as in :class:`returnn.ExtractAudioFeatures`,
  which is used by a couple of datasets, e.g. :class:`returnn.OggZipDataset`.
  """
  # Specify non-none default options:
  _DefaultOpts = dict(
    window_len=0.025, step_len=0.010, features="mfcc")  # as in ExtractAudioFeatures

  def __init__(self, **kwargs):
    self.options = kwargs.copy()
    for key, value in self._DefaultOpts.items():
      self.options.setdefault(key, value)

  def copy(self, **kwargs):
    options = self.options.copy()
    options.update(kwargs)
    return ExtractAudioFeaturesOptions(**options)

  def copy_with_mean_var(self, stats):
    """
    :param AudioOptsAndMeanStddevStatistics stats:
    :rtype: ExtractAudioFeaturesOptions
    """
    return self.copy(norm_mean=stats.mean, norm_std_dev=stats.std_dev)

  def _sis_hash(self):
    return sis_hash_helper(self.options)

  def get_feature_dimension(self):
    """
    :rtype: int
    """
    # Keep logic in sync with returnn.ExtractAudioFeatures.
    if self.options.get("num_feature_filters", None) is None:
      # Set same defaults as in ExtractAudioFeatures.
      if self.options["features"] == "raw":
        num_feature_filters = 1
      else:
        num_feature_filters = 40
    else:
      num_feature_filters = self.options["num_feature_filters"]
    if isinstance(self.options.get("with_delta", None), bool):
      with_delta = 1 if self.options["with_delta"] else 0  # unify for hash
    else:
      with_delta = self.options.get("with_delta", 0)
    join_frames = self.options.get("join_frames", None)
    return (with_delta + 1) * num_feature_filters * (join_frames or 1)

  def as_returnn_data_opts(self):
    """
    :return: opts for :class:`returnn.Data` in :class:`returnn.ExternData`
    :rtype: dict[str]
    """
    dim = self.get_feature_dimension()
    return {"shape": (None, dim), "dim": dim}

  def as_returnn_extract_opts(self):
    """
    :return: opts for :class:`returnn.ExtractAudioFeatures`
    :rtype: dict[str]
    """
    return self.options


class GenericDataset:
  def __init__(self, dependent_paths=None):
    """
    :param list[Path]|None dependent_paths: this is excluded from the hash.
      this is optional, to have explicit dependencies on some (additional) files.
    """
    self.dependent_paths = dependent_paths

  def copy(self, **kwargs):
    assert not kwargs
    from copy import deepcopy
    return deepcopy(self)

  def as_returnn_opts(self):
    """
    :rtype: dict[str]
    """
    raise NotImplementedError


class ExplicitDataset(GenericDataset):
  def __init__(self, returnn_opts, dependent_paths=None):
    """
    :param dict[str] returnn_opts:
    :param list[Path]|None dependent_paths: this is excluded from the hash.
      this is optional, to have explicit dependencies on some (additional) files.
    """
    super(ExplicitDataset, self).__init__(dependent_paths=dependent_paths)
    self.returnn_opts = returnn_opts

  def as_returnn_opts(self):
    """
    :rtype: dict[str]
    """
    return self.returnn_opts


class OggZipDataset(GenericDataset):
  """
  Represents :class:`returnn.OggZipDataset`.
  """

  def __init__(self, name, path, dependent_paths=None, audio_opts=None, targets_opts=None, other_opts=None):
    """
    :param str name: some arbitrary name. used for the hash
    :param str|list[str]|Path path: this is excluded from the hash
    :param list[Path]|None dependent_paths: this is excluded from the hash.
      this is optional, to have explicit dependencies on some files,
      in case that ``path`` is not a :class:`Path`.
    :param ExtractAudioFeaturesOptions|None audio_opts:
    :param dict[str] targets_opts:
    :param dict[str] other_opts: currently not used for hash
    """
    if dependent_paths is None:
      assert isinstance(path, Path)
      dependent_paths = [path]
    super(OggZipDataset, self).__init__(dependent_paths=dependent_paths)
    self.name = name
    self.path = path
    self.audio_opts = audio_opts
    self.targets_opts = targets_opts
    if other_opts is None:
      other_opts = {}
    assert "audio" not in other_opts
    assert "targets" not in other_opts
    self.other_opts = other_opts

  def copy(self, **kwargs):
    """
    :rtype: OggZipDataset
    """
    kwargs = kwargs.copy()
    kwargs.setdefault("name", self.name)
    kwargs.setdefault("path", self.path)
    kwargs.setdefault("dependent_paths", self.dependent_paths)
    kwargs.setdefault("audio_opts", self.audio_opts)
    kwargs.setdefault("targets_opts", self.targets_opts)
    if "other_opts" in kwargs:
      other_opts = kwargs["other_opts"]
      if other_opts is None:
        kwargs["other_opts"] = None
      else:
        new_other_opts = self.other_opts.copy()
        new_other_opts.update(other_opts)
        kwargs["other_opts"] = new_other_opts
    else:
      kwargs["other_opts"] = self.other_opts
    return OggZipDataset(**kwargs)

  def _sis_hash(self):
    d = dict(name=self.name)
    if self.audio_opts:
      d["audio"] = self.audio_opts
    if self.targets_opts:
      d["targets"] = self.targets_opts
    # Note: If we include other_opts, we would probably want to keep it compatible to the old hash.
    # Also, we would probably want to exclude entries like "zip_audio_files_have_name_as_prefix".
    # Also, we intentionally have the name in the hash,
    # so if that name already covers other aspects,
    # we intentionally do not cover these other aspects (e.g. like the path).
    return sis_hash_helper(d)

  def __repr__(self):
    # Somewhat similar logic as in _sis_hash. Only use what's relevant.
    parts = [self.name]
    if self.audio_opts:
      parts.append("audio=%r" % self.audio_opts)
    if self.targets_opts:
      parts.append("targets=%r" % self.targets_opts)
    return "%s<%s>" % (self.__class__.__name__, " ".join(parts))

  def register_output(self, name):
    """
    :param str name:
    """
    for dep_path in self.dependent_paths:
      tk.register_output("%s-%s" % (name, dep_path.path.replace("/", "_")), dep_path)

  def as_returnn_opts(self):
    parts = {
      "class": 'OggZipDataset',
      "path": self.path,
      'use_cache_manager': True,
      "audio": self.audio_opts.options if self.audio_opts else None,
      "targets": self.targets_opts}
    for key, value in sorted(self.other_opts.items()):
      assert key not in parts
      parts[key] = value
    return parts


class ExtractPyTxtBySeqFromOggZipDatasetJob(Job):
  """
  Based on OggZipDataset, extract a txt file in Python format like::

  {seq_tag: raw_txt, ...}
  """
  def __init__(self, dataset):
    """
    :param OggZipDataset dataset:
    """
    # No audio or specific targets needed. We just use the dataset "raw" output.
    # Alternatively, we could also use Utf8ByteTargets.
    # But the RETURNN tool dump-dataset-raw-strings.py currently simply uses "raw".
    self.dataset = self._reduce_dataset(dataset)
    self.output = self.output_path("output-by-seq.txt.gz")

  @classmethod
  def _reduce_dataset(cls, dataset):
    """
    :param OggZipDataset dataset:
    :return: dataset with removed opts which are not needed (e.g. audio feature extraction)
    :rtype: OggZipDataset
    """
    dataset = dataset.copy(audio_opts=None)
    dataset = dataset.copy(targets_opts=None)
    return dataset

  @classmethod
  def hash(cls, parsed_args):
    """
    :param dict[str] parsed_args:
    :rtype: str
    """
    dataset = parsed_args["dataset"]
    assert isinstance(dataset, OggZipDataset)
    dataset = cls._reduce_dataset(dataset)
    return sis_hash(sis_hash_helper(dataset))

  def run(self):
    import subprocess
    args = [
      "%s/returnn/tools/dump-dataset-raw-strings.py" % tk.gs.BASE_DIR,
      "--dataset", repr(self.dataset.as_returnn_opts()),
      "--out", self.output.get_path()]
    print("$ %s" % " ".join(args))
    subprocess.check_call(args)
    assert os.path.exists(self.output.get_path())

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 1}, mini_task=True)


class ExtractTxtFromPyTxtBySeqJob(Job):
  """
  The input txt is in Python format, like::

  {seq_tag: raw_txt, ...}

  e.g. via :class:`ExtractPyTxtBySeqFromOggZipDatasetJob`.

  The output is simple line-based txt format, containing only the raw txt itself.
  """

  def __init__(self, py_txt):
    """
    :param Path py_txt:
    """
    self.py_txt = py_txt
    self.output = self.output_path("output.txt.gz")

  @classmethod
  def hash(cls, parsed_args):
    return sis_hash(sis_hash_helper(parsed_args["py_txt"]))

  def run(self):
    py_txt = eval(generic_open(self.py_txt.get_path()).read())
    assert isinstance(py_txt, dict) and len(py_txt) > 0
    example_key, example_value = next(iter(py_txt.items()))
    assert isinstance(example_key, str) and isinstance(example_value, str)
    with generic_open(self.output.get_path(), "w") as f:
      for seq_tag, raw_txt in sorted(py_txt.items()):
        f.write("%s\n" % raw_txt)

  def tasks(self):
    yield Task('run', rqmt={'mem': 1, 'time': 1}, mini_task=True)


def ogg_zip_dataset_to_txt(dataset):
  """
  :param OggZipDataset dataset:
  :rtype: Path
  """
  py_txt = ExtractPyTxtBySeqFromOggZipDatasetJob(dataset=dataset)
  txt = ExtractTxtFromPyTxtBySeqJob(py_txt=py_txt.output)
  return txt.output


class CreateMeanVarStatisticsFromOggZipDatasetJob(Job):
  """
  Creates mean/std-dev for the audio features.
  """

  def __init__(self, dataset):
    """
    :param OggZipDataset dataset:
    """
    assert isinstance(dataset, OggZipDataset)
    self.dataset = dataset
    assert dataset.audio_opts
    self.output_mean = self.output_path("stats.mean.txt")
    self.output_std_dev = self.output_path("stats.std_dev.txt")
    self._output_common_prefix = self.output_mean.get_path()[:-len(".mean.txt")]
    self.output = AudioOptsAndMeanStddevStatistics(
      dataset=dataset, mean=self.output_mean, std_dev=self.output_std_dev)

  def run(self):
    from pprint import pprint
    import subprocess
    cmd = [
      sys.executable,
      "%s/returnn/tools/dump-dataset.py" % tk.gs.BASE_DIR,
      repr(self.dataset.as_returnn_opts()),
      "--endseq", "-1",
      "--type", "null",
      "--dump_stats", self._output_common_prefix
    ]
    print("Run:")
    pprint(cmd)
    subprocess.check_call(cmd)
    assert os.path.exists(self.output_mean) and os.path.exists(self.output_std_dev)

  def tasks(self):
    yield Task('run', rqmt={'mem': 5, 'time': 20})


class AudioOptsAndMeanStddevStatistics:
  def __init__(self, dataset, mean, std_dev):
    """
    :param OggZipDataset dataset:
    :param Path mean:
    :param Path std_dev:
    """
    self.dataset = dataset
    self.mean = mean
    self.std_dev = std_dev

  def register_output(self, prefix):
    """
    :param str prefix:
    """
    tk.register_output("%s.mean.txt" % prefix, self.mean)
    tk.register_output("%s.std_dev.txt" % prefix, self.std_dev)
