"""
Length reward
"""

from __future__ import annotations
from typing import Any, Tuple
import torch
from ..interface import LabelScorerIntf
from ..interface_dyn_beam import LabelScorerDynBeamIntf


class LengthRewardScorer(LabelScorerIntf):
    """
    Length reward
    """

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
        """initial state"""
        return None

    def score_and_update_state(
        self,
        *,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """score, state"""
        return torch.ones((1, 1, 1), device=prev_label.device), None

    def max_remaining_seq_score(
        self, *, state: Any, max_remaining_steps: torch.Tensor, device: torch.device
    ) -> torch.Tensor:
        """max remaining"""
        return max_remaining_steps.to(torch.get_default_dtype())


class LengthRewardDynBeamScorer(LabelScorerDynBeamIntf):
    """
    Length reward
    """

    def get_initial_state(self, *, batch_size: int, device: torch.device) -> Any:
        """initial state"""
        return None

    def score_and_update_state(
        self,
        *,
        batch_idx: torch.Tensor,
        prev_state: Any,
        prev_label: torch.Tensor,
    ) -> Tuple[torch.Tensor, Any]:
        """score, state"""
        return torch.ones((1, 1), device=prev_label.device), None
