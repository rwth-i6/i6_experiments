"""
Generic interface for decoding.
"""

from __future__ import annotations
from typing import Optional, Any, Dict, Tuple, TYPE_CHECKING
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
            scores: shape [Batch, Beam, Label], log-prob-like scores
            state: all tensors are expected to have shape [Batch, Beam, ...].
        """
        raise NotImplementedError


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


class ShallowFusedLabelScorers(LabelScorerIntf):
    def __init__(self, label_scorers: Dict[str, Tuple[LabelScorerIntf, float]]):
        """
        :param label_scorers: keys are some arbitrary name, value is label scorer and scale for the log-prob
        """
        self.label_scores = label_scorers

    def score_and_update_state(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """score and update state"""
        state = {}
        score = None
        for k, (v, scale) in self.label_scores.items():
            score_, state_ = v.score_and_update_state(prev_state=prev_state[k], prev_label=prev_label)
            state[k] = state_
            score_ = score_ * scale
            score = score_ if score is None else (score_ + score)
        return score, state
