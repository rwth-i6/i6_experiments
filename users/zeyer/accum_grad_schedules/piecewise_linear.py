"""
Piecewise linear
"""


def dyn_accum_grad_piecewise_linear(*, epoch: int, global_train_step: int, **_kwargs) -> int:
    epoch  # unused # noqa
    from returnn.config import get_global_config

    config = get_global_config()

    steps = config.int_list("accum_grad_piecewise_steps")
    values = config.int_list("accum_grad_piecewise_values")
    assert len(steps) + 1 == len(values)

    last_step = 0
    for i, step in enumerate(steps):
        assert step > last_step
        assert global_train_step >= last_step
        if global_train_step < step:
            factor = (global_train_step + 1 - last_step) / (step - last_step)
            return round(values[i + 1] * factor + values[i] * (1 - factor))
        last_step = step

    return values[-1]
