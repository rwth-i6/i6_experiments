from typing import Dict
from i6_experiments.users.berger.args.jobs.recognition_args import (
    get_atr_search_parameters,
    get_atr_lookahead_options,
)
from i6_experiments.users.berger.util import recursive_update
from sisyphus import tk


def get_hybrid_am_args(cart_file: tk.Path, **kwargs) -> Dict:
    am_args = {
        "state_tying": "cart",
        "state_tying_file": cart_file,
    }
    am_args.update(kwargs)
    return am_args


def get_hybrid_train_step_args(**kwargs) -> Dict:
    default_args = {
        "time_rqmt": 168,
        "mem_rqmt": 16,
        "log_verbosity": 5,
    }
    return recursive_update(default_args, kwargs)


def get_hybrid_recog_step_args(num_classes: int, **kwargs) -> Dict:
    default_args = {
        "epochs": ["best"],
        "prior_scales": [0.7],
        "pronunciation_scales": [0.0],
        "lm_scales": [16.0],
        "num_classes": num_classes,
        "lm_lookahead": True,
        "lookahead_options": get_atr_lookahead_options(),
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": get_atr_search_parameters(bp=12.0, bpl=100_000, wep=0.5, wepl=25_000),
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": True,
            "best_path_algo": "bellman-ford",
        },
        "prior_args": {
            "mem_rqmt": 16,
            "time_rqmt": 24,
        },
        "use_gpu": False,
        "rtf": 10,
        "mem": 16,
    }

    return recursive_update(default_args, kwargs)
