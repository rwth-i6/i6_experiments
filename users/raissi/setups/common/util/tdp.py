__all__ = ["format_tdp", "Float", "TDP"]

import typing

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase


Float = typing.Union[float, tk.Variable, DelayedBase]
TDP = typing.Union[Float, str]


def format_tdp_val(val) -> str:
    return "inf" if val == "infinity" else f"{val}"


def format_tdp(tdp) -> str:
    return ",".join(format_tdp_val(v) for v in tdp)
