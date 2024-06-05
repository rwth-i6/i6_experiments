"""
Recog definition
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Protocol, Tuple

if TYPE_CHECKING:
    from returnn.tensor import Tensor, Dim

from .model import ModelT


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
    ) -> Tuple[Tensor, Tensor, Dim, Dim]:
        """
        :return:
            recog results including beam {batch, beam, out_spatial},
            log probs {batch, beam},
            out_spatial_dim,
            final beam_dim
        """
        raise NotImplementedError

    output_with_beam: bool = True  # False not really supported...
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
