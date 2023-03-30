from sisyphus import Path
from i6_experiments.users.berger.args.jobs.search_types import SearchTypes
from typing import Any, Union, Optional, Dict, List


def get_recognition_args(search_type: SearchTypes, **kwargs) -> Dict:
    rec_args = {
        SearchTypes.AdvancedTreeSearch: get_advanced_tree_search_recognition_args,
        SearchTypes.GenericSeq2SeqSearchJob: get_generic_seq2seq_search_recognition_args,
        SearchTypes.ReturnnSearch: get_returnn_search_recognition_args,
    }[search_type](**kwargs)
    rec_args["search_type"] = search_type
    return rec_args


def get_advanced_tree_search_recognition_args(
    *,
    epochs: Optional[List[int]] = None,
    prior_scales: Union[float, List[float]] = 0.3,
    pronunciation_scales: Union[float, List[float]] = 6.0,
    lm_scales: Union[float, List[float]] = 20.0,
    use_gpu: bool = False,
    **kwargs,
) -> Dict[str, Any]:

    if isinstance(prior_scales, float):
        prior_scales = [prior_scales]
    if isinstance(pronunciation_scales, float):
        pronunciation_scales = [pronunciation_scales]
    if isinstance(lm_scales, float):
        lm_scales = [lm_scales]
    epochs = epochs or []

    return {
        "epochs": epochs,
        # "feature_flow_key": "gt",
        "prior_scales": prior_scales,
        "pronunciation_scales": pronunciation_scales,
        "lm_scales": lm_scales,
        "lm_lookahead": True,
        "lookahead_options": get_lookahead_options(**kwargs),
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": get_atr_search_parameters(**kwargs),
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": True,
            "best_path_algo": "bellman-ford",
        },
        "use_gpu": use_gpu,
        "rtf": 50,
        "mem": 8,
    }


def get_returnn_search_recognition_args(
    *,
    epochs: Optional[List[int]] = None,
    prior_scales: Union[float, List[float]] = 0.3,
    use_gpu: bool = True,
    log_prob_layer: str = "output",
) -> Dict[str, Any]:

    if isinstance(prior_scales, float):
        prior_scales = [prior_scales]
    epochs = epochs or []

    return {
        "epochs": epochs,
        "prior_scales": prior_scales,
        "log_prob_layer": log_prob_layer,
        "use_gpu": use_gpu,
        "rtf": 10,
        "mem": 8,
    }


def get_generic_seq2seq_search_recognition_args(
    *,
    epochs: Optional[List[int]] = None,
    prior_scales: Union[float, List[float]] = 0.3,
    lm_scales: Union[float, List[float]] = 0.8,
    add_eow: bool = True,
    add_sow: bool = False,
    recog_lexicon: Optional[Path] = None,
    label_unit: str = "phoneme",
    label_scorer_type: str = "precomputed-log-posterior",
    label_scorer_args: Dict = {},
    label_tree_args: Dict = {},
    use_gpu: bool = False,
    **kwargs,
) -> Dict[str, Any]:

    if isinstance(prior_scales, float):
        prior_scales = [prior_scales]
    if isinstance(lm_scales, float):
        lm_scales = [lm_scales]
    if epochs is None:
        epochs = []

    return {
        "epochs": epochs,
        "prior_scales": prior_scales,
        "lm_scales": lm_scales,
        "lm_lookahead": True,
        "lookahead_options": get_lookahead_options(**kwargs),
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": get_seq2seq_search_parameters(**kwargs),
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": True,
            "best_path_algo": "bellman-ford",
        },
        "label_unit": label_unit,
        "label_file_blank": kwargs.get("allow_blank", True),
        "recog_lexicon": recog_lexicon,
        "add_eow": add_eow,
        "add_sow": add_sow,
        "label_scorer_type": label_scorer_type,
        "label_scorer_args": label_scorer_args,
        "label_tree_args": label_tree_args,
        "use_gpu": use_gpu,
        "rtf": 50,
        "mem": 8,
    }


def get_seq2seq_search_parameters(
    lp: float = 22.0,
    lpl: int = 500000,
    wep: float = 0.5,
    wepl: int = 10000,
    allow_blank: bool = True,
    allow_loop: bool = True,
    blank_penalty: float = 0.0,
    recombination_limit: int = -1,
    fs_decoding: bool = False,
    create_lattice: bool = True,
    optimize_lattice: bool = True,
    prune_trace: bool = False,
    tp: float = 0.5,
    tpl: int = 2000,
    **kwargs,
):
    search_params = {
        "allow-blank-label": allow_blank,
        "allow-label-loop": allow_loop,
        "allow-label-recombination": True,
        "allow-word-end-recombination": True,
        "label-pruning": lp,
        "label-pruning-limit": lpl,
        "word-end-pruning": wep,
        "word-end-pruning-limit": wepl,
        "create-lattice": create_lattice,
        "optimize-lattice": optimize_lattice,
    }

    if recombination_limit > 0:
        search_params["label-recombination-limit"] = recombination_limit

    if blank_penalty:
        search_params["blank-label-penalty"] = blank_penalty

    if prune_trace:
        search_params["trace-pruning"] = tp
        search_params["trace-pruning-limit"] = tpl

    if fs_decoding:
        search_params["full-sum-decoding"] = True
        search_params["label-full-sum"] = True

    return search_params


def get_atr_search_parameters(
    bp: float = 16.0,
    bpl: int = 500000,
    wep: float = 0.5,
    wepl: int = 10000,
    lsp: Optional[float] = None,
    **kwargs,
):
    search_params = {
        "beam-pruning": bp,
        "beam-pruning-limit": bpl,
        "word-end-pruning": wep,
        "word-end-pruning-limit": wepl,
    }
    if lsp is not None:
        search_params["lm-state-pruning"] = lsp
    return search_params


def get_lookahead_options(
    scale: Optional[float] = None,
    hlimit: int = 1,
    clow: int = 2000,
    chigh: int = 3000,
    **kwargs,
):
    lmla_options = {
        "history_limit": hlimit,
        "cache_low": clow,
        "cache_high": chigh,
    }
    if scale is not None:
        lmla_options["scale"] = scale

    return lmla_options
