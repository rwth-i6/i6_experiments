import numpy as np
from typing import Optional

def controlled_noam(
        warmup_epochs: int,
        reduction_epochs: int,
        peak_lr:float,
        min_lr: float,
        warmup_min_lr: Optional[float] = None
):
    """
    This function replicates noam, but with self defined peak_lr and min_lr

    :param warmup_epochs: number of linearly increasing epochs, starting with warmuo_min_lr in epoch one and
        peak_lr in epoch "warmup_epochs"
    :param reduction_epochs: number of square root reduction epochs,
        with the last epoch having exactly min_lr as learning rate
    :param peak_lr: maximum learning rate
    :param min_lr: minimum learning rate
    :param warmup_min_lr: minimum learning rate for linear warmup, if None equal to min_lr
    """
    if warmup_min_lr is None:
        warmup_min_lr = min_lr
    warmup = list(np.linspace(warmup_min_lr, peak_lr, warmup_epochs))
    epoch_scaling = (((peak_lr/min_lr)**2)-1)/reduction_epochs
    reduction = [peak_lr*((1 + (epoch + 1)*epoch_scaling)**-0.5) for epoch in range(reduction_epochs)]
    return warmup + reduction