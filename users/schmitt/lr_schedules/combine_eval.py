"""
Any eval string.
Can combine multiple learning rates schedulers by an eval-string
in the form ``"a * b + c"``,
giving ``{"a": scheduler1, "b": scheduler2, "c": scheduler3}``.
Any local vars, if they are callable, are called first and given the lr func kwargs.
Additionally, you can use ``np``.
"""

from __future__ import annotations


def dyn_lr_combine_eval(*, global_train_step: int, epoch: int, learning_rate: float, **_kwargs) -> float:
    """
    Eval string, e.g. to combine multiple, see module docstring.
    """
    from returnn.config import get_global_config
    import numpy as np

    config = get_global_config()

    eval_str = config.value("learning_rate_eval", None)
    assert eval_str, "learning_rate_eval not specified in config"
    assert isinstance(eval_str, str), f"learning_rate_eval must be str, got {type(eval_str)}"

    def _map_user_eval_local_value(v):
        if callable(v):
            return v(**lr_func_kwargs)
        return v

    eval_locals = config.typed_value("learning_rate_eval_locals")
    if not eval_locals:
        eval_locals = {}
    assert isinstance(eval_locals, dict), "learning_rate_eval_locals must be dict"
    lr_func_kwargs = {"global_train_step": global_train_step, "epoch": epoch, "learning_rate": learning_rate, **_kwargs}
    eval_locals = {k: _map_user_eval_local_value(v) for (k, v) in eval_locals.items()}
    eval_locals.update(lr_func_kwargs)
    eval_locals.update({"np": np})
    res = eval(eval_str, eval_locals)
    assert isinstance(res, float), f"learning_rate_eval {eval_str!r} must return float, got {res!r} ({type(res)})"
    return res
