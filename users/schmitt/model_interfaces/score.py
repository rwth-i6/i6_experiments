"""
Score/rescoring definition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, Optional, Tuple

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim

from .model import ModelT


__all__ = ["RescoreDef"]


class RescoreDef(Protocol[ModelT]):
    """
    Get scores for some target hyps.

    Conceptional very similar to training,
    but we don't use it for training,
    we don't get the scores via mark_as_loss but instead directly returned from the function,
    and we have a beam dim in the targets
    (because the targets come from some earlier beam search).
    """

    def __call__(
        self,
        *,
        model: ModelT,
        data: Optional[Tensor] = None,
        data_spatial_dim: Optional[Dim] = None,
        targets: Tensor,
        targets_spatial_dim: Dim,
        targets_beam_dim: Dim,
    ) -> Tensor:
        """
        :param model:
        :param data: inputs to some encoder. {batch,data_spatial_dim,[feat]}. not given if this is only a LM
        :param data_spatial_dim: dyn size usually of shape {batch}
        :param targets: {batch,targets_beam_dim,targets_spatial_dim}
        :param targets_spatial_dim: dyn size usually of shape {batch,targets_beam_dim}
        :param targets_beam_dim:
        :return: log probs {batch,targets_beam_dim}
        """
        raise NotImplementedError
