"""
Training definition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim

from .model import ModelT


class TrainDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Dim,
        targets: Tensor,
        targets_spatial_dim: Dim,
    ):
        raise NotImplementedError

    learning_rate_control_error_measure: Optional[str] = None


class FramewiseTrainDef(Protocol[ModelT]):
    """
    Defines the losses (mark_as_loss).
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Dim,
        align_targets: Tensor,
        align_targets_spatial_dim: Dim,
    ):
        raise NotImplementedError

    learning_rate_control_error_measure: Optional[str] = None
