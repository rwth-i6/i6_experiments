import torch
from dataclasses import dataclass

from i6_models.config import ModelConfiguration


@dataclass
class IdentityConfig(ModelConfiguration):
    pass


class IdentityModule(torch.nn.Module):
    def __init__(self, _: IdentityConfig):
        super().__init__()

    def forward(self, *args, **kwargs):
        if args and kwargs:
            return (*args, *(kwargs.values()))
        if args:
            return args[0] if len(args) == 1 else args
        if kwargs:
            return next(iter(kwargs.values())) if len(kwargs) == 1 else tuple(kwargs.values())
        return None
