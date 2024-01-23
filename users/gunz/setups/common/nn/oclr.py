__all__ = ["get_oclr_config"]

import math
import typing


# This function is designed by Wei Zhou
def get_learning_rates(
    lrate=0.001,
    pretrain=0,
    warmup=False,
    warmupRatio=0.1,
    increase=90,
    incMinRatio=0.01,
    incMaxRatio=0.3,
    constLR=0,
    decay=90,
    decMinRatio=0.01,
    decMaxRatio=0.3,
    expDecay=False,
    reset=False,
):
    # example fine tuning: get_learning_rates(lrate=5e-5, increase=0, constLR=150, decay=60, decMinRatio=0.1, decMaxRatio=1)
    # example for n epochs get_learning_rates(increase=n/2, cdecay=n/2)
    learning_rates = []
    # pretrain (optional warmup)
    if warmup:
        learning_rates += warmup_lrates(initial=lrate * warmupRatio, final=lrate, epochs=pretrain)
    else:
        learning_rates += [lrate] * pretrain
    # linear increase and/or const
    if increase > 0:
        step = lrate * (incMaxRatio - incMinRatio) / increase
        for i in range(1, increase + 1):
            learning_rates += [lrate * incMinRatio + step * i]
        if constLR > 0:
            learning_rates += [lrate * incMaxRatio] * constLR
    elif constLR > 0:
        learning_rates += [lrate] * constLR
    # linear decay
    if decay > 0:
        if expDecay:  # expotential decay (closer to newBob)
            import numpy as np

            factor = np.exp(np.log(decMinRatio / decMaxRatio) / decay)
            for i in range(1, decay + 1):
                learning_rates += [lrate * decMaxRatio * (factor**i)]
        else:
            step = lrate * (decMaxRatio - decMinRatio) / decay
            for i in range(1, decay + 1):
                learning_rates += [lrate * decMaxRatio - step * i]
    # reset and default newBob(cv) afterwards
    if reset:
        learning_rates += [lrate]
    return learning_rates


# designed by Wei Zhou
def warmup_lrates(initial=0.0001, final=0.001, epochs=20):
    lrates = []
    step_size = (final - initial) / (epochs - 1)
    for i in range(epochs):
        lrates += [initial + step_size * i]
    return lrates


def get_oclr_config(num_epochs: int, *, schedule: str = "v6") -> typing.Dict[str, typing.Any]:
    """Returns learning rate RETURNN config for OneCycle LR."""

    import numpy as np

    def oclr_cfg(lrate: float):
        n = int((num_epochs // 10) * 9)
        n_rest = num_epochs - n
        lrates = get_learning_rates(lrate=lrate, increase=n // 2, decay=math.ceil(n / 2))
        lrates += list(np.linspace(lrates[-1], min([*lrates, 1e-6]), n_rest))

        assert len(lrates) == num_epochs

        return {
            "learning_rate_file": "lr.log",
            "min_learning_rate": 1e-6,
            "learning_rates": lrates,
            "learning_rate_control": "constant",
        }

    peak_lrs = {
        "v6": 0.0003,
        "v7": 0.00053,
        "v8": 0.0003,
        "v9": 0.0004,
        "v10": 0.0006,
        "v11": 0.0007,
        "v12": 0.0008,
        "v13": 0.001,
        "v14": 0.0012,
        "v15": 0.0015,
        "v16": 0.0018,
        "v17": 0.002,
    }

    if not schedule in peak_lrs:
        raise ValueError(f"unknown LR schedule {schedule}")

    return oclr_cfg(lrate=peak_lrs[schedule] / 0.3)
