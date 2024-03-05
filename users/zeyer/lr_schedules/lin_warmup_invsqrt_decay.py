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


def demo(
    warmup_steps: int = 20_000,
    invsqrt_norm: int = 20_000,
    learning_rate: float = 0.0025,
    num_steps: int = 1_000_000,
):
    """demo"""
    from returnn.config import Config, global_config_ctx

    config = Config(
        dict(
            learning_rate_warmup_steps=warmup_steps,
            learning_rate_invsqrt_norm=invsqrt_norm,
        )
    )

    import numpy

    steps = numpy.arange(num_steps)
    with global_config_ctx(config):
        lrs = [dyn_lr_lin_warmup_invsqrt_decay(global_train_step=i, learning_rate=learning_rate) for i in steps]
    lrs = numpy.array(lrs)

    import matplotlib.pyplot as plt

    plt.plot(steps, lrs)
    plt.xlabel("global train step")
    plt.ylabel("learning rate")
    plt.title(
        f"lr={learning_rate}"
        f", warmup={warmup_steps}, invsqrt_norm={invsqrt_norm}"
        f"\nfirst lr={lrs[0]:.4e}, last lr={lrs[-1]:.4e}"
    )
    plt.show()


if __name__ == "__main__":
    demo()
