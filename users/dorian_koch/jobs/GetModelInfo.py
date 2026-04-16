from typing import Any, Optional
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import _PtCheckpoint
from i6_experiments.users.zeyer.train_v4 import dict_update_deep
from sisyphus import Job, Task, tk
from i6_core.returnn.training import PtCheckpoint


class GetModelInfo(Job):
    def __init__(self, model: PtCheckpoint):
        self.model = model
        self.out_num_params = self.output_path("num_params.txt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import torch

        a = torch.load(self.model.path, map_location="cpu")

        num_params = sum(p.numel() for p in a["model"].values())
        with open(self.out_num_params, "w") as f:
            f.write(str(num_params))
