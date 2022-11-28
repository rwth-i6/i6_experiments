__all__ = [
    "RecognitionParameters",
    "PriorPath",
    "SearchJobArgs",
    "Lattice2CtmArgs",
    "ScliteScorerArgs",
    "OptimizeJobArgs",
]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypedDict, Union

from sisyphus import tk

import i6_core.rasr as rasr

from ..config.am_config import Tdp


class RecognitionParameters:
    def __init__(
        self,
        *,
        am_scales: List[float],
        lm_scales: List[float],
        prior_scales: List[float],
        pronunciation_scales: List[Optional[float]],
        tdp_scales: List[float],
        speech_tdps: List[Type[Tdp]],
        silence_tdps: List[Type[Tdp]],
        nonspeech_tdps: List[Optional[Type[Tdp]]],
    ):
        self.am_scales = am_scales
        self.lm_scales = lm_scales
        self.prior_scales = prior_scales
        self.pronunciation_scales = pronunciation_scales
        self.tdp_scales = tdp_scales
        self.speech_tdps = speech_tdps
        self.silence_tdps = silence_tdps
        self.nonspeech_tdps = nonspeech_tdps

    def _get_iter(self):
        return [
            (am, lm, pri, pron, tdp, speech, sil, nonspeech)
            for am, lm, pri, pron, tdp, speech, sil, nonspeech in zip(
                self.am_scales,
                self.lm_scales,
                self.prior_scales,
                self.pronunciation_scales,
                self.tdp_scales,
                self.speech_tdps,
                self.silence_tdps,
                self.nonspeech_tdps,
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
class PriorPath:
    acoustic_mixture_path: tk.Path
    prior_xml_path: tk.Path
    priori_scale: float


class LookaheadOptions(TypedDict):
    test: str


class SearchJobArgs(TypedDict):
    use_gpu: bool
    rtf: float
    mem: int
    cpu: int
    extra_config: Optional[rasr.RasrConfig]
    extra_post_config: Optional[rasr.RasrConfig]


class AdvTreeSearchJobArgs(SearchJobArgs):
    feature_scorer: Type[rasr.FeatureScorer]
    search_parameters: Dict[str, Any]
    lm_lookahead: bool
    lookahead_options: Optional[Union[LookaheadOptions, Dict]]
    create_lattice: bool
    eval_single_best: bool
    eval_best_in_lattice: bool
    lmgc_mem: int


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
