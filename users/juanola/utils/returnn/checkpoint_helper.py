from typing import Set

from returnn.util.math import next_power_of_two


def default_returnn_keep_epochs(num_epochs: int) -> Set[int]:
    """
    Default keep_epochs in RETURNN when cleanup_old_models is enabled
    but "keep" is not specified.
    Excluding the keep_last_n logic.
    See RETURNN cleanup_old_models code.
    """
    from itertools import count

    default_keep_pattern = set()
    if num_epochs <= 10:
        keep_every = 4
        keep_doubles_of = 5
    elif num_epochs <= 50:
        keep_every = 20
        keep_doubles_of = 5
    elif num_epochs <= 100:
        keep_every = 40
        keep_doubles_of = 10
    else:
        keep_every = 80 * next_power_of_two(1 + num_epochs // 240)
        keep_doubles_of = 20

    for i in count(1):
        n = keep_every * i
        if n > num_epochs:
            break
        default_keep_pattern.add(n)
    for i in count():
        n = keep_doubles_of * (2 ** i)
        if n > num_epochs:
            break
        default_keep_pattern.add(n)

    return default_keep_pattern
