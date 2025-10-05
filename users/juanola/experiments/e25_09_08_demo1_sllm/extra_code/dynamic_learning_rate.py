from typing import Optional


def dyn_lr_piecewise_linear(
    *, global_train_step: int, learning_rate: float, epoch_continuous: Optional[float] = None, **_kwargs
) -> float:
    """
    Piecewise linear
    """
    from returnn.config import get_global_config
    from returnn.util.math import PiecewiseLinear

    config = get_global_config()
    f = config.typed_dict.get("_learning_rate_piecewise_cache")
    if f is None:
        steps = config.float_list("learning_rate_piecewise_steps")
        lrs = config.float_list("learning_rate_piecewise_values")
        assert len(steps) + 1 == len(lrs)
        last_step = 0
        for i, step in enumerate(steps):
            assert step > last_step
            last_step = step
        f = PiecewiseLinear(dict(zip([0] + list(steps), lrs)))
        config.typed_dict["_learning_rate_piecewise_cache"] = f
    if config.bool("learning_rate_piecewise_by_epoch_continuous", False):
        assert epoch_continuous is not None
        return f(epoch_continuous) * learning_rate
    return f(global_train_step + 1) * learning_rate
