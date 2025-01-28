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
