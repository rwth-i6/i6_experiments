"""
Generic interfaces to define models, training and recognition.
"""

from __future__ import annotations
from typing import Protocol, TypeVar, Optional
from returnn.tensor import Tensor, Dim
from returnn_common import nn as _nn

# For backwards compatibility with code importing from here.
from .model_with_checkpoints import *  # noqa

ModelT = TypeVar("ModelT", bound=_nn.Module)


class ModelDef(Protocol[ModelT]):
    """
    Creates the model, per epoch
    """

    def __call__(self, *, epoch: int, in_dim: Dim, target_dim: Dim, training: bool = False) -> ModelT:
        raise NotImplementedError

    behavior_version: int
    # Version 2 (i.e. attrib might not be available in all cases):
    backend: Optional[str] = None
    batch_size_factor: int  # according to some arbitrary reference point


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


class RecogDef(Protocol[ModelT]):
    """
    Defines the recog. It returns the recog output.
    Thus, this includes all the recog details, such as beam size, etc.
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Tensor,
        data_spatial_dim: Dim,
    ) -> Tensor:
        """
        :return: recog output, including beam or not, depending on output_with_beam
        """
        raise NotImplementedError

    output_with_beam: bool = True
    output_blank_label: Optional[str] = None

    # A batched beam search can be dependent on the batch size,
    # when the max out seq len depends on the max input seq len in a batch,
    # as we commonly use it for our AED models or RNN-T models.
    # For RNA, the out seq len is always fixed (same as encoder seq len),
    # so there it should not have an effect,
    # and you should set this to False.
    # In any case, the effect should be low,
    # so you might want to set it to False in any case.
    # If you set this here to True,
    # it makes the hash dependent on the batch size.
    batch_size_dependent: bool
