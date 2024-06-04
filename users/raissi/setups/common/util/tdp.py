__all__ = ["to_tdp", "format_tdp", "Float", "TDP"]

from typing import Union, Tuple

from sisyphus import tk
from sisyphus.delayed_ops import DelayedBase, DelayedGetItem

from i6_experiments.common.setups.rasr.config.am_config import Tdp
from i6_experiments.users.raissi.setups.common.data.typings import TDP


def to_tdp(tdp_tuple: Tuple[TDP, TDP, TDP, TDP]) -> Tdp:
    return Tdp(loop=tdp_tuple[0], forward=tdp_tuple[1], skip=tdp_tuple[2], exit=tdp_tuple[3])


def format_tdp_val(val) -> str:
    if isinstance(val, DelayedGetItem):
        val = val.get()
    return "inf" if val == "infinity" else f"{val}"


def format_tdp(tdp) -> str:
    return ",".join(format_tdp_val(v) for v in tdp)
