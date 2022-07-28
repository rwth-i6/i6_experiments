
"""
Switchboard 300h dataset, intended mostly for RWTH i6 internal use.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import os

from returnn_common.datasets.interface import DatasetConfig, VocabConfig


class _Bpe(VocabConfig):
  def __init__(self, dim, codes: str, vocab: str):
    super(_Bpe, self).__init__()
    self.dim = dim
    self.codes = codes
    self.vocab = vocab

  def get_num_classes(self) -> int:
    """
    Get num classes
    """
    return self.dim

  def get_opts(self) -> Dict[str, Any]:
    """
    Get opts
    """
    return {
      'bpe_file': self.codes,
      'vocab_file': self.vocab,
      'unknown_label': None  # should not be needed
      # 'seq_postfix': [0]  # no EOS needed for RNN-T
    }


bpe1k = _Bpe(
  dim=1030,
  codes='/work/asr3/irie/data/switchboard/subword_clean/ready/swbd_clean.bpe_code_1k',
  vocab='/work/asr3/irie/data/switchboard/subword_clean/ready/vocab.swbd_clean.bpe_code_1k')


class SwitchboardExternSprint(DatasetConfig):
  """
  This uses hardcoded paths to i6 internal features.

  Also this is mostly intended for end-to-end setups, where feature extraction always goes through RETURNN.
  For hybrid NN-HMM, you need to be careful.
  E.g. sprint_interface_dataset_opts need to match, using correct `input_stddev` and maybe `bpe`.
  """
  def __init__(self, *,
               vocab: Optional[VocabConfig] = None,
               train_epoch_split=6):
    super(SwitchboardExternSprint, self).__init__()
    self.vocab = vocab
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
    d = {
        "data": {"dim": 40},  # Gammatone 40-dim
    }
    if self.vocab:
        target = "orth_classes"
        d[target] = {"dim": self.vocab.get_num_classes(), "vocab": self.vocab.get_opts(), "sparse": True}
    return d

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

  def get_dataset(self, data: str):
    """
    Get dataset
    """
    from returnn_common.cache_manager import cf
    assert data in {"train", "devtrain", "cv", "dev", "hub5e_01", "rt03s"}
    epoch_split = {"train": self.train_epoch_split}.get(data, 1)
    corpus_name = {"cv": "train", "devtrain": "train"}.get(data, data)  # train, dev, hub5e_01, rt03s

    # TODO fix relative paths (dependencies, RASR config)
    files = {
        "config": "config/training.config",
        "corpus": "/work/asr3/irie/data/switchboard/corpora/%s.corpus.gz" % corpus_name}
    if data in {"train", "cv", "devtrain"}:
        files["segments"] = "dependencies/seg_%s" % {
            "train": "train", "cv": "cv_head3000", "devtrain": "train_head3000"}[data]
    files["features"] = "/u/tuske/work/ASR/switchboard/feature.extraction/gt40_40/data/gt.%s.bundle" % corpus_name
    for k, v in sorted(files.items()):
        assert os.path.exists(v), "%s %r does not exist" % (k, v)
    estimated_num_seqs = {"train": 227047, "cv": 3000, "devtrain": 3000}  # wc -l segment-file

    args = [
        "--config=" + files["config"],
        lambda: "--*.corpus.file=" + cf(files["corpus"]),
        lambda: "--*.corpus.segments.file=" + (cf(files["segments"]) if "segments" in files else ""),
        lambda: "--*.feature-cache-path=" + cf(files["features"]),
        "--*.log-channel.file=/dev/null",
        "--*.window-size=1",
    ]
    args += [
        "--*.corpus.segment-order-shuffle=true",
        "--*.segment-order-sort-by-time-length=true",
        "--*.segment-order-sort-by-time-length-chunk-size=%i" % {"train": epoch_split * 1000}.get(data, -1),
    ]
    d = {
        "class": "ExternSprintDataset", "sprintTrainerExecPath": "sprint-executables/nn-trainer",
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
