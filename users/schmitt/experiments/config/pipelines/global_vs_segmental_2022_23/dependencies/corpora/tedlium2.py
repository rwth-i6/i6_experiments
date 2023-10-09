from sisyphus import *

from i6_core.corpus.convert import CorpusToStmJob


class TedLium2Corpora:
  def __init__(self):
    self.oggzip_paths = {
      "train": [],
      "cv": [],
    }
    self.oggzip_paths["devtrain"] = self.oggzip_paths["train"]

    self.segment_paths = {
      "train": None,
      "cv": None
    }

    self.stm_jobs = {}

    self.stm_paths = {}

    self.corpus_keys = ("train", "cv", "devtrain")
    self.train_corpus_keys = ("train", "cv", "devtrain")

    self.partition_epoch = None
