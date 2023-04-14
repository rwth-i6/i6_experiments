__all__ = ["SearchParameters", "LookaheadOptions", "LatticeToCtmArgs", "NnRecogArgs", "KeyedRecogArgsType"]

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict, Union

from sisyphus import tk

import i6_core.returnn as returnn

# Attribute names are invalid identifiers, therefore use old syntax
SearchParameters = TypedDict(
    "SearchParameters",
    {
        "beam-pruning": float,
        "beam-pruning-limit": float,
        "lm-state-pruning": Optional[float],
        "word-end-pruning": float,
        "word-end-pruning-limit": float,
    },
)


class LookaheadOptions(TypedDict):
    cache_high: Optional[int]
    cache_low: Optional[int]
    history_limit: Optional[int]
    laziness: Optional[int]
    minimum_representation: Optional[int]
    tree_cutoff: Optional[int]


class LatticeToCtmArgs(TypedDict):
    best_path_algo: Optional[str]
    encoding: Optional[str]
    extra_config: Optional[Any]
    extra_post_config: Optional[Any]
    fill_empty_segments: Optional[bool]


class NnRecogArgs(TypedDict):
    acoustic_mixture_path: Optional[tk.Path]
    checkpoints: Optional[Dict[int, returnn.Checkpoint]]
    create_lattice: Optional[bool]
    epochs: Optional[List[int]]
    eval_best_in_lattice: Optional[bool]
    eval_single_best: Optional[bool]
    feature_flow_key: str
    lattice_to_ctm_kwargs: Optional[LatticeToCtmArgs]
    lm_lookahead: bool
    lm_scales: List[float]
    lookahead_options: Optional[LookaheadOptions]
    mem: int
    name: str
    optimize_am_lm_scale: bool
    parallelize_conversion: Optional[bool]
    prior_scales: List[float]
    pronunciation_scales: List[float]
    returnn_config: Optional[returnn.ReturnnConfig]
    rtf: int
    search_parameters: Optional[SearchParameters]
    use_gpu: Optional[bool]


@dataclass()
class NnRecogArgs:
    name: str
    returnn_config: returnn.ReturnnConfig
    checkpoints: Dict[int, returnn.Checkpoint]
    acoustic_mixture_path: tk.Path
    prior_scales: List[float]
    pronunciation_scales: List[float]
    lm_scales: List[float]
    optimize_am_lm_scale: bool
    feature_flow_key: str
    search_parameters: Dict
    lm_lookahead: bool
    lattice_to_ctm_kwargs: Dict
    parallelize_conversion: bool
    rtf: int
    mem: int
    lookahead_options: Optional[Dict] = None
    epochs: Optional[List[int]] = None
    native_ops: Optional[List[str]] = None


# TODO merge the two NnRecogArgs

KeyedRecogArgsType = Dict[str, Union[Dict[str, Any], NnRecogArgs]]
