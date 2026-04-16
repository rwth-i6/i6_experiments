"""
Cosine annealing learning rate schedule with linear warmup and optional constant learning rate phase.
"""

from __future__ import annotations
from typing import Any, Dict
import functools


def linear_warmup_cosine_annealing(
    *,
    learning_rate: float,
    epoch_continuous: float,
    num_warmup_epochs: float,
    max_lr: float,
    min_lr: float,
    num_epochs: int,
    num_constant_epochs: int = 0,
    **_kwargs,
) -> float:
    import math

    if epoch_continuous < num_warmup_epochs:
        """Linear warmup."""
        lr = max_lr * epoch_continuous / num_warmup_epochs
        return learning_rate * lr
    if num_warmup_epochs <= epoch_continuous < num_warmup_epochs + num_constant_epochs:
        """Constant learning rate."""
        return learning_rate * max_lr
    epoch_continuous -= num_warmup_epochs + num_constant_epochs
    """Cosine annealing learning rate schedule."""
    lr = min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * epoch_continuous / num_epochs))
    return learning_rate * lr


def get_linear_warmup_cosine_annealing_cfg(
    n_ep: int,
    *,
    base_lr: float = 1.0,
    num_warmup_epochs: float,
    num_constant_epochs: int = 0,
    max_lr: float,
    min_lr: float,
) -> Dict[str, Any]:
    """
    LR scheduling for RETURNN config
    """
    return {
        "__num_epochs": n_ep,
        "learning_rate": base_lr,
        "dynamic_learning_rate": functools.partial(
            linear_warmup_cosine_annealing,
            num_epochs=n_ep,
            num_warmup_epochs=num_warmup_epochs,
            num_constant_epochs=num_constant_epochs,
            max_lr=max_lr,
            min_lr=min_lr,
        ),
    }
