import torch
from .base import BaseTransform


class FwdScaledBwdId(BaseTransform):
    """
    Forward pass: return scaled
    Backward pass: pass-through (identity) for gradients.
    """

    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = scale

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + (logits * (self.scale - 1)).detach()
