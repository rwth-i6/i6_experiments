from __future__ import annotations
from typing import Any, Dict, Tuple, Optional
from dataclasses import dataclass
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
          prev_align_label: Optional[torch.Tensor] = None,
          t: Optional[int] = None,
  ) -> Tuple[torch.Tensor, Any]:
    """
    :param prev_state: state of the scorer (decoder). any nested structure.
        all tensors are expected to have shape [Batch, Beam, ...].
    :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
        :param prev_align_label: shape [Batch, Beam] -> index in [0...AlignLabel-1]
    :param t: current time step
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
          # prev_align_label and t are currently only used for time-sync scorers and are ignored otherwise
          # TODO: maybe add separate interface for time-sync scorers
          prev_align_label: Optional[torch.Tensor] = None,
          t: Optional[int] = None,
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
    :param prev_align_label: shape [Batch, Beam] -> index in [0...AlignLabel-1]
    :param t: current time step
    :return: (seq_scores_exp, state).
        seq_scores_exp: shape [Batch, Beam, Label], log-prob-like scores for the partial sequence.
            This is basically prev_seq_scores[:,:,None] + scores.
        individual_scores: shape [Batch, Beam, Label], log-prob-like scores.
            Broadcasting allowed in any dim.
            Keys are arbitrary, used for reporting.
        state: all tensors are expected to have shape [Batch, Beam, ...].
    """
    scores, state = self.score_and_update_state(
      prev_state=prev_state, prev_label=prev_label, prev_align_label=prev_align_label, t=t)
    return prev_seq_scores[:, :, None] + scores, {"main": scores}, state


class ShallowFusedLabelScorers(LabelScorerIntf):
  def __init__(self, label_scorers: Dict[str, Tuple[LabelScorerIntf, float]] = None):
    """
    :param label_scorers: keys are some arbitrary name, value is label scorer and scale for the log-prob
    """
    self.label_scorers: Dict[str, Tuple[LabelScorerIntf, float]] = label_scorers or {}

  def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
    """initial state"""
    state = {}
    for k, (v, scale) in self.label_scorers.items():
      state[k] = v.get_initial_state(batch_size=batch_size, device=device)
    return state

  def score_and_update_state(
          self,
          *,
          prev_state: Any,
          prev_label: torch.Tensor,
          prev_align_label: Optional[torch.Tensor] = None,
          t: Optional[int] = None,
  ) -> Tuple[torch.Tensor, Any]:
    """score and update state"""
    state = {}
    score = None
    for k, (v, scale) in self.label_scorers.items():
      score_, state_ = v.score_and_update_state(
        prev_state=prev_state[k], prev_label=prev_label, prev_align_label=prev_align_label, t=t)
      state[k] = state_
      score_ = score_ * scale
      score = score_ if score is None else (score + score_)
    return score, state

  def seq_score_ext_and_update_state(
          self,
          prev_seq_scores: torch.Tensor,
          *,
          prev_state: Any,
          prev_label: torch.Tensor,
          prev_align_label: Optional[torch.Tensor] = None,
          t: Optional[int] = None,
  ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
    """seq score ext and update state"""
    individual_scores = {}
    state = {}
    seq_score = prev_seq_scores[:, :, None]  # [Batch,Beam,1]
    seq_score_ext = None  # [Batch,Beam,Label]
    for k, (v, scale) in self.label_scorers.items():
      score_, state_ = v.score_and_update_state(
        prev_state=prev_state[k], prev_label=prev_label, prev_align_label=prev_align_label, t=t)

      state[k] = state_
      individual_scores[k] = score_
      if scale != 1:
        score_ = score_ * scale
      if score_.shape[-1] == 1:  # broadcast over label
        seq_score = seq_score + score_
      else:
        seq_score_ext = score_ if seq_score_ext is None else (seq_score_ext + score_)
    seq_score_ext = seq_score if seq_score_ext is None else (seq_score + seq_score_ext)
    return seq_score_ext, individual_scores, state


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
