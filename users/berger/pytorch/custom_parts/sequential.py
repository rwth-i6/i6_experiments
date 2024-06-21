from typing import List, Tuple
import torch
from dataclasses import dataclass

from i6_models.config import ModelConfiguration, ModuleFactoryV1


@dataclass
class SequentialModuleV1Config(ModelConfiguration):
    submodules: List[ModuleFactoryV1]

    def __post_init__(self) -> None:
        super().__post_init__()

        assert len(self.submodules) > 0


class SequentialModuleV1(torch.nn.Module):
    def __init__(self, model_cfg: SequentialModuleV1Config):
        super().__init__()

        self.submodules = torch.nn.ModuleList([submodule() for submodule in model_cfg.submodules])

    def forward(self, *args):
        x = args
        for module in self.submodules:
            if isinstance(x, (tuple, list)):
                x = module(*x)
            else:
                x = module(x)
        return x
