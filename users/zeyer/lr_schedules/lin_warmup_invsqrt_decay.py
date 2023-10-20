"""
Linear warmup, inv sqrt decay

Like this:
https://github.com/espnet/espnet/blob/93dafc3806e1b00067dc07c8afe540ff7813b327/espnet2/schedulers/warmup_lr.py
"""


def dyn_lr_lin_warmup_invsqrt_decay(*, global_train_step: int, learning_rate: float, **_kwargs) -> float:
    """
    Linear warmup, inv sqrt decay
    """
    from returnn.config import get_global_config

    config = get_global_config()

    warmup_steps = config.int("learning_rate_warmup_steps", None)
    assert warmup_steps and warmup_steps > 0, f"set learning_rate_warmup_steps in config >0, got {warmup_steps}"

    # In ESPnet, inv_norm_factor = warmup_steps, but we allow it to be configured separately.
    norm = config.float("learning_rate_invsqrt_norm", None)
    assert norm and norm > 0, f"set learning_rate_invsqrt_norm in config >0, got {norm}"

    i = global_train_step + 1
    if i <= warmup_steps:
        return learning_rate * (i / warmup_steps)  # linear warmup
    base = 1.0 + (i - warmup_steps) / norm
    return learning_rate * base**-0.5
