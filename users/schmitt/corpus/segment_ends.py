import zipfile
import copy
from typing import Dict, Any

from sisyphus import Path, Job, Task

from i6_core.returnn.config import ReturnnConfig
from i6_core.lib import corpus


class AugmentCorpusSegmentEndsJob(Job):
  def __init__(self, bliss_corpous: Path, oggzip_path: Path, corpus_key: str = "dev-other"):
    self.bliss_corpus = bliss_corpous
    self.oggzip_path = oggzip_path
    self.corpus_key = corpus_key

    self.out_bliss_corpus = self.output_path("corpus.xml.gz")

  def tasks(self):
    yield Task("run", rqmt={"cpu": 1, "mem": 4, "time": 1, "gpu": 0})

  def run(self):
    corpus_ = corpus.Corpus()
    corpus_.load(self.bliss_corpus.get_path())

    with zipfile.ZipFile(self.oggzip_path.get_path(), "r") as zip_ref:
      for file in zip_ref.namelist():
        if file == "out.ogg.txt":
          with zip_ref.open(file) as f:
            oggzip_segment_list = eval(f.read())

    durations = {}
    for segment in oggzip_segment_list:
      durations[segment["seq_name"]] = segment["duration"]

    for segment in corpus_.segments():
      assert segment.start == 0.0
      segment.end = durations[f"{self.corpus_key}/{segment.name}/{segment.name}"]

    corpus_.dump(self.out_bliss_corpus.get_path())

  @classmethod
  def hash(cls, kwargs: Dict[str, Any]):
    d = copy.deepcopy(kwargs)

    if d["corpus_key"] == "dev-other":
      d.pop("corpus_key")

    return super().hash(d)
