import torch
from .base import BaseTransform


class FwdZeroBwdId(BaseTransform):
    """
    Forward pass: return zero (i.e. uniform distribution).
    Backward pass: pass-through (identity) for gradients.
    """

    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        return logits + (-logits).detach()  # zero, but grads will go to logits
