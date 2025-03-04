from typing import Dict, Iterator, Any, Tuple
import shutil

from sisyphus import *

from i6_core.returnn.training import PtCheckpoint


class GetCheckpointWithBestWer(Job):
  def __init__(self, checkpoint_to_wer: Dict[Tuple[Any, PtCheckpoint], tk.Variable]):
    self.checkpoint_to_wer = checkpoint_to_wer
    self.out_checkpoint = self.output_path("checkpoint")
    self.out_summary = self.output_path("summary")

  def tasks(self) -> Iterator[Task]:
    yield Task("run", mini_task=True)

  def run(self):
    best_checkpoint = min(self.checkpoint_to_wer, key=lambda x: self.checkpoint_to_wer[x].get())
    shutil.copy(best_checkpoint[1].path, self.out_checkpoint.get_path())

    with open(self.out_summary, "w") as f:
      for checkpoint, wer in self.checkpoint_to_wer.items():
        f.write(f"{checkpoint[0]}: {wer.get()}\n")


class GetWerForSegments(Job):
  def __init__(
          self,
          sclite_pra_path: tk.Path,
          segment_path: tk.Path,
  ):
    self.sclite_pra_path = sclite_pra_path
    self.segment_path = segment_path

    self.out_wer = self.output_var("wer")
    self.out_num_segments = self.output_var("num_segments")

  def tasks(self) -> Iterator[Task]:
    yield Task("run", mini_task=True)

  def run(self):
    filtered_s = 0
    filtered_d = 0
    filtered_i = 0
    filtered_c = 0

    with open(self.segment_path.get_path(), "r") as f:
      segments = f.readlines()
      segments = [segment.split("/")[-1].strip() for segment in segments]

    segment = None
    with open(self.sclite_pra_path.get_path(), "r") as f:
      for line in f:
        if line.startswith("File:"):
          segment = line.split()[-1].strip()
        elif line.startswith("Scores:"):
          assert segment is not None
          if segment not in segments:
            continue
          c, s, d, i = map(int, line.split("#I)")[1].strip().split())
          filtered_s += s
          filtered_d += d
          filtered_i += i
          filtered_c += c
          segment = None

    with open(self.out_wer.get_path(), "w") as f:
      f.write(f"{(filtered_s + filtered_d + filtered_i) / (filtered_s + filtered_d + filtered_i + filtered_c)}")

    with open(self.out_num_segments.get_path(), "w") as f:
      f.write(str(len(segments)))
