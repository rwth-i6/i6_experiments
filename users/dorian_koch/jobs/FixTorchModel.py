from typing import Any, Optional
from i6_experiments.users.zeyer.model_interfaces import ModelWithCheckpoint
from i6_experiments.users.zeyer.model_interfaces.model_with_checkpoints import _PtCheckpoint
from i6_experiments.users.zeyer.train_v4 import dict_update_deep
from sisyphus import Job, Task, tk
from i6_core.returnn.training import PtCheckpoint


class FixTorchModelJob(Job):
    def __init__(self, model: PtCheckpoint, fix_func):
        self.model = model
        self.fix_func = fix_func
        self.out_model = self.output_path("model.pt")

    def tasks(self):
        yield Task("run", mini_task=True)

    def run(self):
        import torch
        a = torch.load(self.model.path, map_location="cpu")

        assert "model" in a, "model key not found in checkpoint"
    
        a = self.fix_func(a)

        torch.save(a, self.out_model)

    def get_out_checkpoint(self, old_checkpoint: ModelWithCheckpoint):
        return ModelWithCheckpoint(
            old_checkpoint.definition,
            (
                _PtCheckpoint(
                    self.out_model
                )
            ),
        )

def update_prefix(model_dict: dict, updates: list[tuple[str, Optional[str]]]) -> dict:
    assert "model" in model_dict, "model key not found in checkpoint"

    old_model: dict[str, Any] = model_dict["model"]
    assert isinstance(old_model, dict), "model should be a dict"
    new_model = dict()
    for k, v in old_model.items():
        for old_prefix, new_prefix in updates:
            if k.startswith(old_prefix):
                if new_prefix is None:
                    break # this key should be removed
                new_key = new_prefix + k[len(old_prefix):]
                new_model[new_key] = v
                break

    new_model_dict = {k: v for k, v in model_dict.items() if k != "model"}
    new_model_dict["model"] = new_model
    return new_model_dict