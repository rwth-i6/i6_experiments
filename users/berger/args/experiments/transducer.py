import copy
from typing import Dict

from i6_experiments.users.berger.args.jobs.recognition_args import (
    get_seq2seq_lookahead_options,
    get_seq2seq_search_parameters,
)
from i6_experiments.users.berger.util import recursive_update

transducer_recog_am_args = {
    "state_tying": "monophone",
    "states_per_phone": 1,
    "tdp_scale": 1.0,
    "tdp_transition": (0.0, 0.0, "infinity", 0.0),
    "tdp_silence": (0.0, 0.0, "infinity", 0.0),
    "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
    "phon_history_length": 0,
    "phon_future_length": 0,
}


def get_transducer_train_step_args(**kwargs) -> Dict:
    default_args = {
        "time_rqmt": 168,
        "mem_rqmt": 16,
        "log_verbosity": 5,
    }
    return recursive_update(default_args, kwargs)


def get_transducer_recog_step_args(
    num_classes: int, reduction_subtrahend: int = 1039, reduction_factor: int = 640, **kwargs
) -> Dict:
    default_args = {
        "epochs": ["best"],
        "lm_scales": [0.7],
        "prior_scales": [0.0],
        "am_args": copy.deepcopy(transducer_recog_am_args),
        "use_gpu": False,
        "label_scorer_type": "tf-ffnn-transducer",
        "label_scorer_args": {
            "use_prior": False,
            "num_classes": num_classes,
            "extra_args": {
                "blank_label_index": 0,
                "reduction_subtrahend": reduction_subtrahend,
                "reduction_factors": reduction_factor,
                "context_size": 1,
                "max_batch_size": 256,
                "use_start_label": True,
                "start_label_index": num_classes,
                "transform_output_negate": True,
            },
        },
        "label_tree_args": {
            "use_transition_penalty": False,
            "skip_silence": True,
        },
        "lookahead_options": get_seq2seq_lookahead_options(),
        "search_parameters": get_seq2seq_search_parameters(
            lp=18.0,
            allow_blank=True,
            allow_loop=False,
        ),
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": True,
        },
        "prior_args": {
            "mem_rqmt": 16,
        },
        "rtf": 50,
        "mem": 8,
    }

    return recursive_update(default_args, kwargs)


def get_transducer_align_step_args(num_classes: int, reduction_factor: int = 4, **kwargs) -> Dict:
    default_args = {
        "epochs": ["best"],
        "prior_scales": [0.3],
        "use_gpu": False,
        "am_args": transducer_recog_am_args,
        "alignment_options": {
            "label-pruning": 50,
            "label-pruning-limit": 100000,
        },
        "align_node_options": {
            "allow-label-loop": False,
        },
        "label_scorer_type": "tf-ffnn-transducer",
        "label_scorer_args": {
            "use_prior": True,
            "num_classes": num_classes,
            "extra_args": {
                "blank_label_index": 0,
                "reduction_factors": reduction_factor,
            },
        },
    }

    return recursive_update(default_args, kwargs)
