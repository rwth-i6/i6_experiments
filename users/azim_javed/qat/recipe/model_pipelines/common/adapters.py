from dataclasses import dataclass

import torch
from i6_models.config import ModelConfiguration


@dataclass
class ErrorAdapterConfig(ModelConfiguration):
    location: str

    def __post_init__(self):
        super().__post_init__()
        if self.location != "output_log_probs":
            raise ValueError(f"Unsupported error-adapter location: {self.location!r}")


class IdentityInitializedLinear(torch.nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        if dimension <= 0:
            raise ValueError(f"Adapter dimension {dimension}")
        self.linear = torch.nn.Linear(dimension, dimension)
        with torch.no_grad():
            torch.nn.init.eye_(self.linear.weight)
            torch.nn.init.zeros_(self.linear.bias)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(tensor)
