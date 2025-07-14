"""
Attribution (saliency/grads, or grad*input, etc.) reduction functions.
"""

from __future__ import annotations
from typing import Protocol
from functools import partial
import torch


def get_attr_reduce_func(name: str) -> AttrReductionFunc:
    """
    :param name: "sum" or "L{p}"
    :return: func
    """
    if name == "sum":
        return sum
    elif name.startswith("L"):
        p = float(name[1:])
        return partial(norm, p=p)
    else:
        raise ValueError(f"Unknown attr reduction function: {name!r}")


class AttrReductionFunc(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: [B,T,F]
        :return: [B,T], reduced F
        """


# noinspection PyShadowingBuiltins
def sum(x: torch.Tensor) -> torch.Tensor:
    """
    :param x: [B,T,F]
    :return: [B,T], reduced F
    """
    return x.sum(dim=-1)


def norm(x: torch.Tensor, *, p: float) -> torch.Tensor:
    """
    :param x: [B,T,F]
    :param p: norm order
    :return: [B,T], reduced F
    """
    return x.norm(p=p, dim=-1)
