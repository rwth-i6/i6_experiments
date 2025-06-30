"""
Generic interface for decoding.
"""

from __future__ import annotations
from typing import Any, Dict, Tuple, TYPE_CHECKING
from dataclasses import dataclass

if TYPE_CHECKING:
    import torch


class LabelScorerIntf:
    """
    Gives a score per label.

    (This could be an output label, or also alignment label, including blank.)

    This object is intended to be recreated (or updated) for every new batch/input.

    If there is some encoder output, or any precomputed output to be reused inside,
    this can be stored as attrib, e.g. via __init__.

    ``state`` is any nested structure (to be used with PyTorch _pytree or similar),
    and allows the following leaf types:

    * :class:`torch.Tensor`
    * :class:`StateObjTensorExt`
    * :class:`StateObjIgnored`
    * ``None``

    Here we assume a fixed beam size for all entries in the batch.
    For dynamic beam sizes, see :class:`.interface_dyn_beam.LabelScorerDynBeamIntf`.
    """

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
        """
        :param batch_size:
        :param device:
        :return: state. all tensors are expected to have shape [Batch, Beam=1, ...].
        """
        raise NotImplementedError

    def score_and_update_state(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch, Beam, ...].
        :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
        :return: (scores, state).
            scores: shape [Batch, Beam, Label], log-prob-like scores.
                Broadcasting is allowed for any of the dims (e.g. think of :class:`LengthRewardScorer`).
            state: all tensors are expected to have shape [Batch, Beam, ...].
        """
        raise NotImplementedError

    def seq_score_ext_and_update_state(
        self,
        prev_seq_scores: torch.Tensor,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
        """
        This provides a default implementation,
        but we might be able to do it more efficient in some cases
        like :class:`ShallowFusedLabelScorers` with broadcasting in scores.

        :param prev_seq_scores: shape [Batch, Beam], log-prob-like scores for the prev partial sequence.
            Broadcasting is allowed in any dim.
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch, Beam, ...].
        :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
        :return: (seq_scores_exp, state).
            seq_scores_exp: shape [Batch, Beam, Label], log-prob-like scores for the partial sequence.
                This is basically prev_seq_scores[:,:,None] + scores.
            individual_scores: shape [Batch, Beam, Label], log-prob-like scores.
                Broadcasting allowed in any dim.
                Keys are arbitrary, used for reporting.
            state: all tensors are expected to have shape [Batch, Beam, ...].
        """
        scores, state = self.score_and_update_state(prev_state=prev_state, prev_label=prev_label)
        return prev_seq_scores[:, :, None] + scores, {"main": scores}, state

    def max_remaining_seq_score(
        self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """
        :param state: current state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch, Beam, ...].
        :param max_remaining_steps: [Batch, Beam|1] (int). how many steps this scorer will potentially run further.
            tensor expected to be on device.
        :param device:
        :return: [Batch|1, Beam|1] (float, like other scores).
            what maximum sum of scores we can get from remaining frames
        """
        raise OptionalNotImplementedError


@dataclass
class StateObjTensorExt:
    """
    Can be used in ``state`` to represent a tensor with additional objects attached.
    The additional objects will be ignored in transformations.
    """

    tensor: torch.Tensor
    extra: Any


@dataclass
class StateObjIgnored:
    """
    Can be used in ``state`` to wrap some other object, which will be ignored in transformations.
    """

    content: Any


class OptionalNotImplementedError(NotImplementedError):
    """optional"""
