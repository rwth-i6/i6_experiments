__all__ = [
    "BaseRecognitionParameters",
    "PriorArgs",
    "SearchJobArgs",
    "Lattice2CtmArgs",
    "ScorerArgs",
]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from sisyphus import tk

from ..config.am_config import TransitionTdp, SilenceTdp


class BaseRecognitionParameters(TypedDict):
    am_scales: List[float]
    lm_scales: List[float]
    pronunciation_scales: List[float]
    transition_tdp: List[TransitionTdp]
    silence_tdp: List[SilenceTdp]


@dataclass()
class PriorArgs:
    acoustic_mixture_path: tk.Path
    prior_xml_path: tk.Path
    scale: float


class SearchJobArgs(TypedDict):
    search_params: Dict[str, Any]


class Lattice2CtmArgs(TypedDict):
    best_path_algo: str


class ScorerArgs(TypedDict):
    cer: bool
    sort_files: bool
    additional_args: Optional[List[str]]
