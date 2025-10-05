from typing import Any, Dict

from i6_experiments.users.juanola.lr_schedules.piecewise_linear import dyn_lr_piecewise_linear


def get_cfg_lrlin_oclr_by_bs_nep_v4(
    n_ep: int,
    *,
    base_lr: float = 1.0,
    peak_lr: float = 1e-3,
    low_lr: float = 1e-5,
    lowest_lr: float = 1e-6,
    step_peak_fraction: float = 0.45,
    step_finetune_fraction: float = 0.9,
) -> Dict[str, Any]:
    """
    :param n_ep: num epochs
    """
    return {
        "learning_rate": base_lr,
        "dynamic_learning_rate": dyn_lr_piecewise_linear,
        "learning_rate_piecewise_by_epoch_continuous": True,
        "learning_rate_piecewise_steps": [step_peak_fraction * n_ep, step_finetune_fraction * n_ep, n_ep],
        "learning_rate_piecewise_values": [low_lr, peak_lr, low_lr, lowest_lr],
    }
