import numpy as np


def get_newbob_cfg(min_learning_rate: float = 1e-6, warmup: bool = True):
    assert min_learning_rate > 0

    base = {
        "learning_rate_file": "lr.log",
        "min_learning_rate": min_learning_rate,
    }

    return {
        **base,
        "learning_rates": list(np.linspace(3e-4, 8e-4, 10)) if warmup else [],
        "learning_rate_control": "newbob_multi_epoch",
        "learning_rate_control_min_num_epochs_per_new_lr": 3,
        "learning_rate_control_relative_error_relative_lr": True,
        "newbob_learning_rate_decay": 0.8,
        "newbob_multi_num_epochs": 40,
        "newbob_multi_update_interval": 1,
    }
