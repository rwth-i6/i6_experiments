"""
Piecewise linear
"""


def dyn_lr_piecewise_linear(*, global_train_step: int, learning_rate: float, **_kwargs) -> float:
    """
    Piecewise linear
    """
    from returnn.config import get_global_config

    config = get_global_config()

    steps = config.int_list("learning_rate_piecewise_steps")
    lrs = config.float_list("learning_rate_piecewise_values")
    assert len(steps) + 1 == len(lrs)

    last_step = 0
    for i, step in enumerate(steps):
        assert step > last_step
        assert global_train_step >= last_step
        if global_train_step < step:
            factor = (global_train_step - last_step) / (step - last_step)
            return learning_rate * (lrs[i + 1] * factor + lrs[i] * (1 - factor))
        last_step = step

    return learning_rate * lrs[-1]
