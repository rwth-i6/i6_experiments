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
            factor = (global_train_step + 1 - last_step) / (step - last_step)
            return learning_rate * (lrs[i + 1] * factor + lrs[i] * (1 - factor))
        last_step = step

    return learning_rate * lrs[-1]


def test_piecewise_linear():
    from numpy.testing import assert_almost_equal, assert_equal

    def _f(x, xs, ys):
        assert isinstance(x, int)
        assert len(xs) + 1 == len(ys)
        last_step = 0
        for i, step in enumerate(xs):
            assert isinstance(step, int)
            assert step > last_step
            assert x >= last_step
            if x < step:
                factor = (x + 1 - last_step) / (step - last_step)
                return ys[i + 1] * factor + ys[i] * (1 - factor)
            last_step = step

        return ys[-1]

    assert_almost_equal(_f(0, [10, 20], [0, 1, 0.5]), 0.1)
    assert_almost_equal(_f(5, [10, 20], [0, 1, 0.5]), 0.6)
    assert_equal(_f(9, [10, 20], [0, 1, 0.5]), 1)
    assert_almost_equal(_f(10, [10, 20], [0, 1, 0.5]), 0.95)
    assert_almost_equal(_f(11, [10, 20], [0, 1, 0.5]), 0.90)
    assert_almost_equal(_f(15, [10, 20], [0, 1, 0.5]), 0.70)
    assert_almost_equal(_f(19, [10, 20], [0, 1, 0.5]), 0.5)
    assert_equal(_f(20, [10, 20], [0, 1, 0.5]), 0.5)
