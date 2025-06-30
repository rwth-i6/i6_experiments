"""
Any object can have some function which returns some label scorer.
"""

from __future__ import annotations
import returnn.frontend as rf
from typing import TYPE_CHECKING, Protocol, Tuple

if TYPE_CHECKING:
    from ..decoding.beam_search_torch.interface import LabelScorerIntf as TorchLabelScorerIntf


class IMakeLabelScorerTorch(Protocol):
    def __call__(self) -> TorchLabelScorerIntf: ...


RFModelWithMakeLabelScorer = Tuple[rf.Module, IMakeLabelScorerTorch]
