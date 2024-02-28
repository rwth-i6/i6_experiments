"""
Generic interface for decoding.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, TYPE_CHECKING
from .interface import StateObjIgnored, StateObjTensorExt

if TYPE_CHECKING:
    import torch


__all__ = ["LabelScorerDynBeamIntf", "StateObjIgnored", "StateObjTensorExt"]


class LabelScorerDynBeamIntf:
    """
    Gives a score per label.

    (This could be an output label, or also alignment label, including blank.)

    This object is intended to be recreated (or updated) for every new batch/input.

    If there is some encoder output, or any precomputed output to be reused inside,
    this can be stored as attrib, e.g. via __init__.

    ``state`` is any nested structure (to be used with PyTorch _pytree or similar),
    and allows the following leaf types:

    * :class:`torch.Tensor`
    * :class:`.interface.StateObjTensorExt`
    * :class:`.interface.StateObjIgnored`
    * ``None``

    The difference to :class:`.interface.LabelScorerIntf` is that here we allow dynamic beam sizes.
    """

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
        """
        :param batch_size: original batch size (Batch).
        :param device:
        :return: state. all tensors are expected to have shape [Batch, ...].
        """
        raise NotImplementedError

    def score_and_update_state(
        self,
        *,
        batch_idx: torch.Tensor,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """
        :param batch_idx: [Batch_] -> batch index of the original batch in [0...Batch-1].
            Batch_ would be [Batch,Beam] packed together (not padded), allowing for dynamic beam sizes.
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch_, ...].
        :param prev_label: shape [Batch_] -> index in [0...Label-1]
        :return: (scores, state).
            scores: shape [Batch_, Label], log-prob-like scores.
                Broadcasting is allowed for any of the dims (e.g. think of :class:`LengthRewardScorer`).
            state: all tensors are expected to have shape [Batch_, ...].
        """
        raise NotImplementedError

    def seq_score_ext_and_update_state(
        self,
        prev_seq_scores: torch.Tensor,
        *,
        batch_idx: torch.Tensor,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
        """
        This provides a default implementation,
        but we might be able to do it more efficient in some cases
        like :class:`ShallowFusedLabelScorers` with broadcasting in scores.

        :param prev_seq_scores: shape [Batch_], log-prob-like scores for the prev partial sequence.
            Broadcasting is allowed in any dim.
        :param batch_idx: [Batch_] -> batch index of the original batch in [0...Batch-1].
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch_, ...].
        :param prev_label: shape [Batch_] -> index in [0...Label-1]
        :return: (seq_scores_exp, state).
            seq_scores_exp: shape [Batch_, Label], log-prob-like scores for the partial sequence.
                This is basically prev_seq_scores[:,None] + scores.
            individual_scores: shape [Batch_, Label], log-prob-like scores.
                Broadcasting allowed in any dim.
                Keys are arbitrary, used for reporting.
            state: all tensors are expected to have shape [Batch_, ...].
        """
        scores, state = self.score_and_update_state(batch_idx=batch_idx, prev_state=prev_state, prev_label=prev_label)
        return prev_seq_scores[:, None] + scores, {"main": scores}, state
