import zipfile
import copy
from typing import Dict, Any

from sisyphus import Path, Job, Task

from i6_core.returnn.config import ReturnnConfig
from i6_core.lib import corpus


class RenameBlissCorpusJob(Job):
  def __init__(self, bliss_corpus, new_name: str):
    self.bliss_corpus = bliss_corpus
    self.new_name = new_name

    self.out_corpus = self.output_path("corpus.xml.gz")

  def tasks(self):
    yield Task("run", mini_task=True)

  def run(self):
    corpus_ = corpus.Corpus()
    corpus_.load(self.bliss_corpus.get_path())

    corpus_.name = self.new_name

    corpus_.dump(self.out_corpus.get_path())
