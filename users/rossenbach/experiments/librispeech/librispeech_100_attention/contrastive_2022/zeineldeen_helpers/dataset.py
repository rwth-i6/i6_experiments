from sisyphus import *
from recipe.sprint import SprintCommand


class Targets:

  def as_returnn_data_opts(self, vocab_size, **kwargs):
    """
    Used for extern_data in RETURNN config
    This require the vocab_size since it is computed by the job that creates the vocabulary file

    :param tk.Variable vocab_size: vocabulary size or number of labels
    :return: opts for :class:`Data` in :class:`ExternData`
    """
    raise NotImplementedError

  def as_returnn_targets_opts(self):
    """
    :return data opts such as vocabulary or bpe
    :rtype: dict[str]
    """
    raise NotImplementedError


class VocabBasedTargets(Targets):

  def __init__(self, vocab, unk_label=None):
    """
    :param Path|None vocab: bpe vocab file path
    :param str unk_label: unknown label
    """
    self.vocab = vocab
    self.unk_label = unk_label

  def as_returnn_data_opts(self, vocab_size, **kwargs):
    """
    :param tk.Variable|int vocab_size: number of labels
    :rtype: dict[str]
    """
    d = {'shape': (None,), 'dim': vocab_size, 'sparse': True}
    d.update(kwargs)
    return d

  def as_returnn_targets_opts(self, **kwargs):
    """
    :rtype: dict[str]
    """
    return {
      "vocab_file": self.vocab,
      "unknown_label": self.unk_label,
      **kwargs
    }


class BpeTargets(VocabBasedTargets):

  def __init__(self, codes, vocab, seq_postfix=0, unk_label=None):
    """
    :param Path codes: bpe codes file path
    :param Path vocab: vocab file path
    :param str unk_label: unknown label
    """
    super(BpeTargets, self).__init__(vocab=vocab, unk_label=unk_label)
    self.codes = codes
    assert isinstance(seq_postfix, int)
    self.seq_postfix = seq_postfix

  def register_outputs(self, prefix):
    tk.register_output('%s.codes' % prefix, self.codes)
    tk.register_output('%s.vocab' % prefix, self.vocab)

  def as_returnn_targets_opts(self):
    opts = {
      'class': 'BytePairEncoding',
      'bpe_file': self.codes,
      'vocab_file': self.vocab,
      'unknown_label': self.unk_label
    }
    if self.seq_postfix is not None:
      opts['seq_postfix'] = [self.seq_postfix]
    return opts


class SentencePieceTargets(VocabBasedTargets):

  def __init__(self, spm_model, extra_opts=None):

    super(SentencePieceTargets, self).__init__(vocab=None)
    self.spm_model = spm_model
    self.extra_opts = extra_opts
    if self.extra_opts is None:
      self.extra_opts = {}

  def as_returnn_targets_opts(self):
    opts = {
      'class': 'SentencePieces',
      'model_file': self.spm_model,
      'add_eos': True,
      **self.extra_opts
    }
    return opts


class AudioFeaturesOpts:
  """
  Encapsulates options for audio features used by OggZipDataset via :class:ExtractAudioFeaturesOptions in RETURNN
  """

  _default_options = dict(window_len=0.025, step_len=0.010, features='mfcc')

  def __init__(self, **kwargs):
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
    feat_dim = self.get_feat_dim()
    return {'shape': (None, feat_dim), 'dim': feat_dim}

  def as_returnn_extract_opts(self):
    return self.options


class GenericDataset:

  def as_returnn_opts(self):
    """
    return data dict for SprintDataset, OggZipDataset, etc
    :return: dict[str]
    """
    raise NotImplementedError


class SprintDataset(GenericDataset):
  """
  Represents :class:`ExternSprintDataset` dataset in RETURNN
  """

  def __init__(self, is_train, name, rasr_train_config, corpus_xml, features, targets_data_opts, segments_file=None,
               epoch_split=1, log_channel=None, other_opts=None):
    """
    :param str|Path rasr_train_config: path to RASR train config (need to for RASR only)
    :param str|Path corpus_xml: path to bliss xml corpus
    :param str|Path segments_file: path to segments file
    :param str|Path features: path to input features basically RASR cache files
    :param dict[str, dict[str]] targets_data_opts: represents targets RETURNN dict, e.g BytePairEncoding class
    :param int epoch_split: data epoch split
    :param dict[str] other_opts: some other options
    """

    self.rasr_train_config = rasr_train_config
    self.corpus_xml = corpus_xml
    self.segments_file = segments_file
    self.features = features
    self.targets_data_opts = targets_data_opts
    self.epoch_split = epoch_split
    if not is_train:
      self.epoch_split = 1
    if log_channel is None:
      log_channel = '{}.sprint.log'.format(name)
    self.log_channel = log_channel
    if other_opts is None:
      other_opts = {}
    self.other_opts = other_opts
    self.seg_order_sort_chunk_size = (self.epoch_split or 1) * 1000 if is_train else -1
    self.trainer_exec = SprintCommand.default_exe('nn-trainer')

  def get_sprint_args(self):
    files = dict()
    files['config'] = tk.uncached_path(self.rasr_train_config)
    files['corpus'] = tk.uncached_path(self.corpus_xml)
    if self.segments_file:
      files['segments'] = tk.uncached_path(self.segments_file)
    files['features'] = tk.uncached_path(self.features)

    args = [
      "--config=" + files["config"],
      "--*.corpus.file=`cf {}`".format(files["corpus"]),
      "--*.corpus.segments.file=" + ("`cf {}`".format(files["segments"]) if 'segments' in files else ''),
      "--*.corpus.segment-order-shuffle=true",
      "--*.segment-order-sort-by-time-length=true",
      "--*.segment-order-sort-by-time-length-chunk-size=%i" % self.seg_order_sort_chunk_size,
      "--*.feature-cache-path=`cf {}`".format(files['features']),
      "--*.log-channel.file=" + self.log_channel,
      "--*.window-size=1",
    ]
    return args

  def as_returnn_opts(self):
    d = {
      'class': 'ExternSprintDataset',
      'sprintTrainerExecPath': self.trainer_exec,
      'sprintConfigStr': self.get_sprint_args(),
      'partitionEpoch': self.epoch_split
    }
    if self.segments_file:
      with open(tk.uncached_path(self.segments_file), 'r') as f:
        d['estimated_num_seqs'] = len(f.readlines()) // self.epoch_split
    # SprintDataset does not accept targets opts with defined class as OggZipDataset, thus they need to be removed
    for k in self.targets_data_opts:
      if 'class' in self.targets_data_opts[k]:
        del self.targets_data_opts[k]['class']
    d.update(self.targets_data_opts)
    d.update(self.other_opts)
    return d


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


class ComputeMeanVarStatisticsFromRETURNNDatasetJob(Job):
  """
  Creates mean/stddev for audio features
  """

  def __init__(self, dataset_opts):
    """
    :param dict[str] dataset_opts: dict representing RETURNN dataset
    """
    self.dataset_opts = dataset_opts
    self.output_mean = self.output_path('stats.mean.txt')
    self.output_stddev = self.output_path('stats.std_dev.txt')
    self.output_prefix = self.output_mean.get_path()[:-len('.mean.txt')]

  def run(self):
    import sys
    cmd = [
      sys.executable,
      '%s/tools/dump-dataset.py' % tk.gs.CRNN_ROOT,
      repr(self.dataset_opts),
      '--endseq', '-1',
      '--type', 'null',
      '--dump_stats', self.output_prefix
    ]
    self.sh(cmd)

  def tasks(self):
    yield Task('run', rqmt={'mem': 5, 'time': 20})


class LibriSpeechDataset(GenericDataset):
  """
  Represents :class:`LibriSpeechCorpus` in RETURNN
  """

  def __init__(self, ogg_path, prefix, audio_opts, target_opts, subset=None, train_epoch_split=20,
               other_opts=None, is_train=True):
    """
    :param str|Path ogg_path: path to ogg zip files
    :param str prefix: e.g train, dev, train-clean, etc
    :param dict[str] audio_opts: represents audio options
    :param dict[str, dict[str]]|None target_opts: represents targets opts with a target key.
      Example: 'bpe': {'bpe_file': ..., ...}
    :param int subset: the number of dataset subset, e.g used for cross-validation
    :param int train_epoch_split: training epoch split
    :param dict[str] other_opts: some other dataset opts
    :param bool is_train: this is needed in case a subset of train is used for evaluation
    """
    self.ogg_path = ogg_path
    self.prefix = prefix
    self.audio_opts = audio_opts
    self.target_opts = target_opts
    self.subset = subset
    self.train_epoch_split = train_epoch_split
    if other_opts is None:
      other_opts = {}
    else:
      other_opts = other_opts.copy()
    self.other_opts = other_opts
    self.is_train = is_train

  def as_returnn_opts(self):
    d = {
      'class': 'LibriSpeechCorpus',
      'path': self.ogg_path,
      "use_zip": True,
      "use_ogg": True,
      "use_cache_manager": True,
      "prefix": self.prefix,
      "audio": self.audio_opts,
      "targets": self.target_opts
    }
    if self.prefix.startswith("train") and self.is_train:
      d["partition_epoch"] = self.train_epoch_split
      if self.prefix == "train":
        d["epoch_wise_filter"] = {
          (1, 5): {
            'use_new_filter': True,
            'max_mean_len': 50,  # chars
            'subdirs': ['train-clean-100', 'train-clean-360']},
          (5, 10): {
            'use_new_filter': True,
            'max_mean_len': 150,  # chars
            'subdirs': ['train-clean-100', 'train-clean-360']},
          (11, 20): {
            'use_new_filter': True,
            'subdirs': ['train-clean-100', 'train-clean-360']},
        }
      num_seqs = 281241  # total
      d["seq_ordering"] = "laplace:%i" % (num_seqs // 1000)
    else:
      d["fixed_random_seed"] = 1
      d["seq_ordering"] = "sorted_reverse"
    if self.subset:
      d["fixed_random_subset"] = self.subset  # faster
    d.update(self.other_opts)
    return d


class MetaDataset(GenericDataset):
  """
  Represents `:class:MetaDataset` in RETURNN
  """

  def __init__(self, data_map, datasets, seq_order_control_dataset, other_opts=None):
    """
    :param list[tuple(str, str)] data_map:
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
