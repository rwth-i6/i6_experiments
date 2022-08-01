__all__ = [
    "BaseRecognitionParameters",
    "PriorArgs",
    "SearchJobArgs",
    "Lattice2CtmArgs",
    "ScliteScorerArgs",
    "OptimizeJobArgs",
]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypedDict

from sisyphus import tk

import i6_core.rasr as rasr

from ..config.am_config import Tdp


class BaseRecognitionParameters:
    def __init__(
        self,
        *,
        am_scales: List[float],
        lm_scales: List[float],
        prior_scales: List[float],
        pronunciation_scales: List[Optional[float]],
        tdp_scales: List[float],
        transition_tdps: List[Type[Tdp]],
        silence_tdps: List[Type[Tdp]],
    ):
        self.am_scales = am_scales
        self.lm_scales = lm_scales
        self.prior_scales = prior_scales
        self.pronunciation_scales = pronunciation_scales
        self.tdp_scales = tdp_scales
        self.transition_tdps = transition_tdps
        self.silence_tdps = silence_tdps

    def _get_iter(self):
        return [
            (am, lm, pron, tdp, trans, sil, pri)
            for am, lm, pri, pron, tdp, trans, sil in zip(
                self.am_scales,
                self.lm_scales,
                self.prior_scales,
                self.pronunciation_scales,
                self.tdp_scales,
                self.transition_tdps,
                self.silence_tdps,
            )
        ]

    def __iter__(self):
        self.iterations = self._get_iter()
        return self

    def __next__(self):
        if self.iterations:
            return self.iterations.pop(0)
        else:
            raise StopIteration


@dataclass()
class PriorArgs:
    acoustic_mixture_path: tk.Path
    prior_xml_path: tk.Path


class SearchJobArgs(TypedDict):
    feature_flow: rasr.FlowNetwork
    feature_scorer: Type[rasr.FeatureScorer]
    search_parameters: Dict[str, Any]
    model_combination_config: Optional[rasr.RasrConfig]


class Lattice2CtmArgs(TypedDict):
    best_path_algo: str
    fill_empty_segments: bool
    encoding: str
    extra_config: Optional[rasr.RasrConfig]
    extra_post_config: Optional[rasr.RasrConfig]


class ScliteScorerArgs(TypedDict):
    ref: Optional[tk.Path]
    cer: bool
    sort_files: bool
    additional_args: Optional[List[str]]
    sctk_binary_path: Optional[tk.Path]


class OptimizeJobArgs(TypedDict):
    opt_only_lm_scale: bool
    maxiter: int
    precision: int
    extra_config: Optional[rasr.RasrConfig]
    extra_post_config: Optional[rasr.RasrConfig]
