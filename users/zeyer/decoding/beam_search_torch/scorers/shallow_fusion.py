"""
Shallow fusion
"""

from __future__ import annotations
from typing import Any, Tuple, Dict
import torch
from ..interface import LabelScorerIntf


class ShallowFusedLabelScorers(LabelScorerIntf):
    def __init__(self, label_scorers: Dict[str, Tuple[LabelScorerIntf, float]] = None):
        """
        :param label_scorers: keys are some arbitrary name, value is label scorer and scale for the log-prob
        """
        self.label_scorers: Dict[str, Tuple[LabelScorerIntf, float]] = label_scorers or {}

    def score_and_update_state(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """score and update state"""
        state = {}
        score = None
        for k, (v, scale) in self.label_scorers.items():
            score_, state_ = v.score_and_update_state(prev_state=prev_state[k], prev_label=prev_label)
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
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Any]:
        """seq score ext and update state"""
        individual_scores = {}
        state = {}
        seq_score = prev_seq_scores[:, :, None]  # [Batch,Beam,1]
        seq_score_ext = None  # [Batch,Beam,Label]
        for k, (v, scale) in self.label_scorers.items():
            score_, state_ = v.score_and_update_state(prev_state=prev_state[k], prev_label=prev_label)
            state[k] = state_
            individual_scores[k] = score_
            score_ = score_ * scale
            if score_.shape[-1] == 1:  # broadcast over label
                seq_score = seq_score + score_
            else:
                seq_score_ext = score_ if seq_score_ext is None else (seq_score_ext + score_)
        seq_score_ext = seq_score if seq_score_ext is None else (seq_score + seq_score_ext)
        return seq_score_ext, individual_scores, state
