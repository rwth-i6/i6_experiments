"""
Length reward
"""

from __future__ import annotations
from typing import Any, Tuple
import torch
from ..interface import LabelScorerIntf


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
