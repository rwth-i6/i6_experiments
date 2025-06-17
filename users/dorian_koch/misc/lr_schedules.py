from __future__ import annotations
from typing import Optional


def dyn_lr_piecewise_loglinear(
    *, global_train_step: int, learning_rate: float, epoch_continuous: Optional[float] = None, **_kwargs
) -> float:
    """
    Piecewise loglinear
    """
    pass
