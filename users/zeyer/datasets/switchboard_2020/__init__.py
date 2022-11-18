
"""
Switchboard 300h dataset, intended mostly for RWTH i6 internal use.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFormat
from i6_core.returnn.config import CodeWrapper
from returnn_common.datasets.base import Dataset
from returnn_common.datasets_old_2022_10.interface import DatasetConfig, VocabConfig
from i6_experiments.users.zeyer import tools_paths

_my_dir = os.path.dirname(os.path.abspath(__file__))
_rasr_configs_dir = _my_dir + "/rasr_configs"


class _Bpe(VocabConfig):
  def __init__(self,
               dim: int,
               codes: str,  # filename
               vocab: str,  # filename
               *,
               eos_idx: Optional[int] = None,
               bos_idx: Optional[int] = None,
               unknown_label: Optional[str] = None,
               other_opts: Optional[Dict[str, Any]] = None,
               ):
    super(_Bpe, self).__init__()
    self.dim = dim
    self.codes = codes
    self.vocab = vocab
    self.eos_idx = eos_idx
    self.bos_idx = bos_idx
    self.unknown_label = unknown_label
    self.other_opts = other_opts

  def get_num_classes(self) -> int:
    """
    Get num classes
    """
    return self.dim

  def get_opts(self) -> Dict[str, Any]:
    """
    Get opts
    """
    d = {
      'bpe_file': self.codes,
      'vocab_file': self.vocab,
      'unknown_label': self.unknown_label,
      'bos_label': self.bos_idx,
      'eos_label': self.eos_idx,
      # 'seq_postfix': [0]  # no EOS needed for RNN-T
    }
    if self.other_opts:
      d.update(self.other_opts)
      if self.other_opts.get("class") == "SamplingBytePairEncoding":
        d.pop("bpe_file")
    return d

  def get_eos_idx(self) -> Optional[int]:
    """EOS"""
    return self.eos_idx

  def get_bos_idx(self) -> Optional[int]:
    """BOS"""
    return self.bos_idx

  def copy(self, **kwargs):
    """Copy"""
    opts = {
      k: getattr(self, k)
      for k in ["dim", "codes", "vocab", "eos_idx", "bos_idx", "unknown_label", "other_opts"]}
    opts.update(kwargs)
    return _Bpe(**opts)


bpe1k = _Bpe(
  dim=1030, eos_idx=0, bos_idx=0,
  codes='/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k',
  vocab='/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k')

bpe1k_with_unk = bpe1k.copy(unknown_label="UNK")


class SwitchboardExternSprint(Dataset):
  """
  This uses hardcoded paths to i6 internal features.

  Also this is mostly intended for end-to-end setups, where feature extraction always goes through RETURNN.
  For hybrid NN-HMM, you need to be careful.
  E.g. sprint_interface_dataset_opts need to match, using correct `input_stddev` and maybe `bpe`.
  """

  # TODO this is incomplete...

  def __init__(self, *,
               vocab: Optional[VocabConfig] = None,
               main_key: Optional[str] = None,
               train_epoch_split=6,
               additional_options: Optional[Dict[str, Any]] = None,
               ):
    super(SwitchboardExternSprint, self).__init__(additional_options=additional_options)
    self.vocab = vocab
    assert main_key in {None, "train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    self.main_key = main_key
    self.train_epoch_split = train_epoch_split

  @classmethod
  def defaults2020(cls,
                   vocab: Optional[VocabConfig] = bpe1k,
                   **kwargs) -> SwitchboardExternSprint:
    """
    Return dataset with old defaults
    """
    return SwitchboardExternSprint(vocab=vocab, **kwargs)

  def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
    """
    Get extern data
    """
    from returnn.tf.util.data import FeatureDim, SpatialDim, batch_dim
    time_dim = SpatialDim("time")
    feature_dim = FeatureDim("audio", 40)  # Gammatone 40-dim
    out_spatial_dim = SpatialDim("out-spatial")
    classes_dim = FeatureDim("vocab", dimension=self.vocab.get_num_classes())
    d = {
        "data": {"dim_tags": [batch_dim, time_dim, feature_dim]},
    }
    if self.vocab:
        target = "orth_classes"
        d[target] = {
            "dim_tags": [batch_dim, out_spatial_dim],
            "sparse_dim": classes_dim,
            "vocab": self.vocab.get_opts()
        }
    return d

  def get_default_target(self) -> Optional[str]:
    """default target"""
    if self.vocab:
      return "orth_classes"
    return None

  def get_train_dataset(self) -> Dict[str, Any]:
    """
    Get train dataset
    """
    return self.get_dataset("train")

  def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
    """
    Get eval datasets
    """
    return {
      "dev": self.get_dataset("cv"),
      "devtrain": self.get_dataset("devtrain")}

  def get_main_name(self) -> str:
    """main key"""
    assert self.main_key, "main key not defined"
    return self.main_key

  def as_returnn_opts(self) -> Dict[str, Any]:
    """main dataset"""
    assert self.main_key, "main key not defined"
    return self.get_dataset(self.main_key)

  def get_dataset(self, data: str) -> Dict[str, Any]:
    """
    Get dataset
    """
    assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": self.train_epoch_split}.get(data, 1)
    corpus_name = {"cv": "train", "devtrain": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

    files = {
        "config": tk.Path(
            f"{_rasr_configs_dir}/merged.config",
            hash_overwrite="switchboard2020/merged.config"),
        "feature_extraction_config": tk.Path(
            f"{_rasr_configs_dir}/base.cache.flow",
            hash_overwrite="switchboard2020/base.cache.flow"),
        "corpus": get_bliss_xml_corpus(corpus_name)}
    if data in {"train", "cv", "devtrain"}:
        filename = "dependencies/seg_%s" % {
            "train": "train", "cv": "cv_head3000", "devtrain": "train_head3000"}[data]
        files["segments"] = tk.Path(
            "/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k/" + filename,
            hash_overwrite="switchboard2020/" + filename,
            cached=True)
    files["features"] = tk.Path(
        "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name,
        hash_overwrite="switchboard2020/tuske-features-gt.%s.bundle" % corpus_name,
        cached=True)
    for k, v in sorted(files.items()):
        assert isinstance(v, tk.Path)
    estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

    args = [
        _DelayedCodeFormat("lambda: '--config=' + cf({!r})", files["config"]),
        _DelayedCodeFormat("lambda: '--*.corpus.file=' + cf({!r})", files["corpus"]),
        _DelayedCodeFormat(
            "lambda: '--*.corpus.segments.file=' + cf({!r})", (files["segments"] if "segments" in files else "")),
        _DelayedCodeFormat("lambda: '--*.feature-cache-path=' + cf({!r})", files["features"]),
        _DelayedCodeFormat("lambda: '--*.feature-extraction.file=' + cf({!r})", files["feature_extraction_config"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    args += [
        "--*.corpus.segment-order-shuffle=true",
        "--*.segment-order-sort-by-time-length=true",
        "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1),
    ]
    d = {
        "class": "ExternSprintDataset",
        "sprintTrainerExecPath": tools_paths.get_rasr_exe("nn-trainer"),
        "sprintConfigStr": args,
        "suppress_load_seqs_print": True,  # less verbose
        "input_stddev": 3.,
        "orth_vocab": self.vocab.get_opts() if self.vocab else None,
    }
    partition_epochs_opts = {
        "partition_epoch": epoch_split,
        "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None,
    }
    d.update(partition_epochs_opts)
    return d


class SwitchboardExternSprintOld(DatasetConfig):
  """
  This uses hardcoded paths to i6 internal features.

  Also this is mostly intended for end-to-end setups, where feature extraction always goes through RETURNN.
  For hybrid NN-HMM, you need to be careful.
  E.g. sprint_interface_dataset_opts need to match, using correct `input_stddev` and maybe `bpe`.
  """

  def __init__(self, *,
               vocab: Optional[VocabConfig] = None,
               main_key: Optional[str] = None,
               train_epoch_split=6):
    super(SwitchboardExternSprintOld, self).__init__()
    self.vocab = vocab
    assert main_key in {None, "train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    self.main_key = main_key
    self.train_epoch_split = train_epoch_split

  @classmethod
  def defaults2020(cls,
                   vocab: Optional[VocabConfig] = bpe1k,
                   **kwargs) -> SwitchboardExternSprintOld:
    """
    Return dataset with old defaults
    """
    return SwitchboardExternSprintOld(vocab=vocab, **kwargs)

  def get_extern_data(self) -> Dict[str, Dict[str, Any]]:
    """
    Get extern data
    """
    from returnn.tf.util.data import FeatureDim, SpatialDim, batch_dim
    time_dim = SpatialDim("time")
    feature_dim = FeatureDim("audio", 40)  # Gammatone 40-dim
    out_spatial_dim = SpatialDim("out-spatial")
    classes_dim = FeatureDim("vocab", dimension=self.vocab.get_num_classes())
    d = {
        "data": {"dim_tags": [batch_dim, time_dim, feature_dim]},
    }
    if self.vocab:
        target = "orth_classes"
        d[target] = {
            "dim_tags": [batch_dim, out_spatial_dim],
            "sparse_dim": classes_dim,
            "vocab": self.vocab.get_opts()
        }
    return d

  def get_default_target(self) -> Optional[str]:
    """default target"""
    if self.vocab:
      return "orth_classes"
    return None

  def get_train_dataset(self) -> Dict[str, Any]:
    """
    Get train dataset
    """
    return self.get_dataset("train")

  def get_eval_datasets(self) -> Dict[str, Dict[str, Any]]:
    """
    Get eval datasets
    """
    return {
      "dev": self.get_dataset("cv"),
      "devtrain": self.get_dataset("devtrain")}

  def get_main_name(self) -> str:
    """main key"""
    assert self.main_key, "main key not defined"
    return self.main_key

  def get_main_dataset(self) -> Dict[str]:
    """main dataset"""
    assert self.main_key, "main key not defined"
    return self.get_dataset(self.main_key)

  def get_dataset(self, data: str) -> Dict[str, Any]:
    """
    Get dataset
    """
    assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": self.train_epoch_split}.get(data, 1)
    corpus_name = {"cv": "train", "devtrain": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

    files = {
        "config": tk.Path(
            f"{_rasr_configs_dir}/merged.config",
            hash_overwrite="switchboard2020/merged.config"),
        "feature_extraction_config": tk.Path(
            f"{_rasr_configs_dir}/base.cache.flow",
            hash_overwrite="switchboard2020/base.cache.flow"),
        "corpus": get_bliss_xml_corpus(corpus_name)}
    if data in {"train", "cv", "devtrain"}:
        filename = "dependencies/seg_%s" % {
            "train": "train", "cv": "cv_head3000", "devtrain": "train_head3000"}[data]
        files["segments"] = tk.Path(
            "/u/zeyer/setups/switchboard/2019-10-22--e2e-bpe1k/" + filename,
            hash_overwrite="switchboard2020/" + filename,
            cached=True)
    files["features"] = tk.Path(
        "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name,
        hash_overwrite="switchboard2020/tuske-features-gt.%s.bundle" % corpus_name,
        cached=True)
    for k, v in sorted(files.items()):
        assert isinstance(v, tk.Path)
    estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

    args = [
        _DelayedCodeFormat("lambda: '--config=' + cf({!r})", files["config"]),
        _DelayedCodeFormat("lambda: '--*.corpus.file=' + cf({!r})", files["corpus"]),
        _DelayedCodeFormat(
            "lambda: '--*.corpus.segments.file=' + cf({!r})", (files["segments"] if "segments" in files else "")),
        _DelayedCodeFormat("lambda: '--*.feature-cache-path=' + cf({!r})", files["features"]),
        _DelayedCodeFormat("lambda: '--*.feature-extraction.file=' + cf({!r})", files["feature_extraction_config"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    args += [
        "--*.corpus.segment-order-shuffle=true",
        "--*.segment-order-sort-by-time-length=true",
        "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1),
    ]
    d = {
        "class": "ExternSprintDataset",
        "sprintTrainerExecPath": tools_paths.get_rasr_exe("nn-trainer"),
        "sprintConfigStr": args,
        "suppress_load_seqs_print": True,  # less verbose
        "input_stddev": 3.,
        "orth_vocab": self.vocab.get_opts() if self.vocab else None,
    }
    partition_epochs_opts = {
        "partition_epoch": epoch_split,
        "estimated_num_seqs": (estimated_num_seqs[data] // epoch_split) if data in estimated_num_seqs else None,
    }
    d.update(partition_epochs_opts)
    return d


def get_bliss_xml_corpus(corpus_name: str) -> tk.Path:
    """Bliss XML"""
    corpus_name = {"hub5e_00": "dev"}.get(corpus_name, corpus_name)
    assert corpus_name in {"dev", "hub5e_01", "rt03s", "train"}
    return tk.Path(
        "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name,
        hash_overwrite="switchboard2020/irie-corpora-%s.corpus.gz" % corpus_name,
        cached=True)


class _DelayedCodeFormat(DelayedFormat):
    """Delayed code"""

    def get(self) -> CodeWrapper:
        """get"""
        return CodeWrapper(super(_DelayedCodeFormat, self).get())
