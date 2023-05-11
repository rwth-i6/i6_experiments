"""
Helpers to create learning rate schedules.
"""
import numpy as np
from typing import List, Optional


def oclr_default_schedule(
        peak_lr: float = 1e-3,
        start_lr: Optional[float] = None,
        end_lr: Optional[float] = None,
        final_lr: float = 1e-8,
        increase_epochs: int = 99,
        peak_epochs: int = 2,
        decrease_epochs: int = 99,
        final_epochs: int = 60,
) -> List[float]:
    """
    :param peak_lr: maximum lr
    :param start_lr: start lr at beginning of the linear cycle
    :param end_lr: end lr at end of the linear cycle
    :param final_lr: final lr after final decay
    :param increase_epochs: number of epochs for increasing lr
    :param peak_epochs: number of epochs at peak lr
    :param decrease_epochs: number of epochs for decreasing lr
    :param final_epochs: number of epochs for final decay
    """
    start_lr = start_lr or peak_lr / 10
    end_lr = end_lr or peak_lr / 10
    lr = (
        list(np.linspace(start_lr, peak_lr, increase_epochs + 1))[:-1] +
        [peak_lr] * peak_epochs +
        list(np.linspace(peak_lr, end_lr, decrease_epochs + 1))[1:] +
        list(np.linspace(end_lr, final_lr, final_epochs))
    )
    return lr
