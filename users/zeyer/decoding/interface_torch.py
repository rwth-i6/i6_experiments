"""
Generic interface for decoding.
"""

from typing import Any, Dict, Tuple
import torch


class LabelScorer:
    """
    Gives a score per label.

    (This could be an output label, or also alignment label, including blank.)

    This object is intended to be recreated (or updated) for every new batch/input.

    If there is some encoder output, or any precomputed output to be reused inside,
    this can be stored as attrib, e.g. via __init__.
    """

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
        """
        :param batch_size:
        :param device:
        :return: state. all tensors are expected to have shape [Batch, Beam=1, ...].
        """
        raise NotImplementedError

    def update_state(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Any:
        """
        :param prev_state: state of the scorer (decoder). any nested structure.
            all tensors are expected to have shape [Batch, Beam, ...].
        :param prev_label: shape [Batch, Beam] -> index in [0...Label-1]
        :return: state. all tensors are expected to have shape [Batch, Beam, ...].
        """
        raise NotImplementedError

    def score(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        """
        :param prev_state: see update_state
        :param prev_label: see update_state
        :param state: output of update_state
        :return: shape [Batch, Beam, Label], log-prob-like scores
        """
        raise NotImplementedError


class ShallowFusedLabelScorers(LabelScorer):
    def __init__(self, label_scorers: Dict[str, Tuple[LabelScorer, float]]):
        """
        :param label_scorers: keys are some arbitrary name, value is label scorer and scale for the log-prob
        """
        self.label_scores = label_scorers

    def update_state(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Any:
        """update state"""
        return {
            k: v.update_state(prev_state=prev_state[k], prev_label=prev_label)
            for k, (v, _) in self.label_scores.items()
        }

    def score(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
        state: Any,
    ) -> torch.Tensor:
        """score"""
        score = None
        for k, (v, scale) in self.label_scores.items():
            score_ = v.score(prev_state=prev_state[k], prev_label=prev_label, state=state[k]) * scale
            score = score_ if score is None else (score_ + score)
        return score
