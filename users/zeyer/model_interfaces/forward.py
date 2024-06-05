"""
Any forwarding
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Tuple

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim

from .model import ModelT


class ForwardDef(Protocol[ModelT]):
    def __call__(self, source: Tensor, /, in_spatial_dim: Dim, model: ModelT) -> Tuple[Tensor, Dim]:
        """
        :param source: input tensor
        :param in_spatial_dim: input spatial dimension
        :param model:
        :return: output tensor and output spatial dimension (can be same as in_spatial_dim)
        """
