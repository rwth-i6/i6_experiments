import torch
from i6_core.returnn import PtCheckpoint
from sisyphus import Job, Task


class ExtractAdapterCheckpointJob(Job):
    def __init__(self, checkpoint: PtCheckpoint):
        self.checkpoint = checkpoint
        self.out_checkpoint = PtCheckpoint(self.output_path("adapter.pt"))

    def tasks(self):
        yield Task("run", rqmt={"cpu": 1, "mem": 12, "time": 1})

    def run(self):
        checkpoint = torch.load(self.checkpoint.path.get_path(), map_location="cpu")
        state = {key: value for key, value in checkpoint["model"].items() if key.startswith("adapters.")}
        if not state:
            raise ValueError("Checkpoint contains no adapter weights")
        torch.save({"model": state}, self.out_checkpoint.path.get_path())
