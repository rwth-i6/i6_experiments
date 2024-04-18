__all__ = [
    "DevRecognitionParameters",
    "TestRecognitionParameters",
    "RecognitionParameters",
    "PriorPath",
    "SearchJobArgs",
    "Lattice2CtmArgs",
    "StmArgs",
    "ScliteScorerArgs",
    "OptimizeJobArgs",
]

from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

from sisyphus import tk

import i6_core.rasr as rasr

from i6_experiments.common.setups.rasr.config.am_config import Tdp


class DevRecognitionParameters:
    def __init__(
        self,
        *,
        am_scales: List[float],
        lm_scales: List[float],
        prior_scales: List[float],
        pronunciation_scales: Optional[List[float]],
        tdp_scales: List[float],
        speech_tdps: List[Tdp],
        silence_tdps: List[Tdp],
        nonspeech_tdps: Optional[List[Tdp]],
        altas: Optional[List[float]],
    ):
        self.am_scales = am_scales
        self.lm_scales = lm_scales
        self.prior_scales = prior_scales
        self.pronunciation_scales = pronunciation_scales if pronunciation_scales is not None else [1.0]
        self.tdp_scales = tdp_scales
        self.speech_tdps = speech_tdps
        self.silence_tdps = silence_tdps
        self.nonspeech_tdps = nonspeech_tdps if nonspeech_tdps is not None else [None]
        self.altas = altas if altas is not None else [0.0]

    def _get_iter(self):
        return [
            (am, lm, pri, pron, tdp, speech, sil, nonspeech, al)
            for am, lm, pri, pron, tdp, speech, sil, nonspeech, al in product(
                self.am_scales,
                self.lm_scales,
                self.prior_scales,
                self.pronunciation_scales,
                self.tdp_scales,
                self.speech_tdps,
                self.silence_tdps,
                self.nonspeech_tdps,
                self.altas,
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


class TestRecognitionParameters:
    def __init__(
        self,
        *,
        am_scales: float,
        lm_scales: float,
        prior_scales: float,
        pronunciation_scales: Optional[float],
        tdp_scales: float,
        speech_tdps: Tdp,
        silence_tdps: Tdp,
        nonspeech_tdps: Optional[Tdp],
        altas: Optional[float],
    ):
        self.am_scales = am_scales
        self.lm_scales = lm_scales
        self.prior_scales = prior_scales
        self.pronunciation_scales = pronunciation_scales if pronunciation_scales is not None else 1.0
        self.tdp_scales = tdp_scales
        self.speech_tdps = speech_tdps
        self.silence_tdps = silence_tdps
        self.nonspeech_tdps = nonspeech_tdps
        self.altas = altas if altas is not None else 0.0


RecognitionParameters = Union[DevRecognitionParameters, TestRecognitionParameters]


@dataclass()
class PriorPath:
    prior_xml_path: tk.Path
    acoustic_mixture_path: tk.Path


class LookaheadOptions(TypedDict):
    test: str


class SearchJobArgs(TypedDict):
    search_parameters: Dict[str, Any]
    use_gpu: bool
    rtf: float
    mem: int
    cpu: int
    extra_config: Optional[rasr.RasrConfig]
    extra_post_config: Optional[rasr.RasrConfig]


class AdvTreeSearchJobArgs(SearchJobArgs):
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
    parallelize: bool
    extra_config: Optional[rasr.RasrConfig]
    extra_post_config: Optional[rasr.RasrConfig]


class StmArgs(TypedDict):
    exclude_non_speech: bool
    non_speech_tokens: Optional[List[str]]
    remove_punctuation: bool
    punctuation_tokens: Optional[Union[str, List[str]]]
    fix_whitespace: bool
    name: str
    tag_mapping: List[Tuple[Tuple[str, str, str], Dict[int, tk.Path]]]


class ScliteScorerArgs(TypedDict):
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
