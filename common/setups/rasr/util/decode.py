__all__ = [
    "BaseRecognitionParameters",
    "PriorArgs",
    "SearchJobArgs",
    "Lattice2CtmArgs",
    "ScorerArgs",
    "OptimizeJobArgs",
]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict

from sisyphus import tk

import i6_core.rasr as rasr

from ..config.am_config import TransitionTdp, SilenceTdp


class BaseRecognitionParameters(TypedDict):
    am_scales: List[float]
    lm_scales: List[float]
    pronunciation_scales: List[float]
    tdp_scales: List[float]
    transition_tdp: List[TransitionTdp]
    silence_tdp: List[SilenceTdp]
    prior_scales: List[float]


@dataclass()
class PriorArgs:
    acoustic_mixture_path: tk.Path
    prior_xml_path: tk.Path


class SearchJobArgs(TypedDict):
    search_params: Dict[str, Any]


class Lattice2CtmArgs(TypedDict):
    best_path_algo: str


class ScorerArgs(TypedDict):
    cer: bool
    sort_files: bool
    additional_args: Optional[List[str]]


class OptimizeJobArgs(TypedDict):
    crp: rasr.CommonRasrParameters
    lattice_cache: Any
