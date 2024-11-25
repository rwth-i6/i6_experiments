import copy
from typing import Dict

from sisyphus import tk

from i6_experiments.users.berger.args.jobs.recognition_args import (
    get_seq2seq_lookahead_options,
    get_seq2seq_search_parameters,
)
from i6_experiments.users.berger.util import recursive_update
from i6_experiments.users.berger.systems.dataclasses import FeatureType
from i6_experiments.users.berger.systems.functors.recognition.returnn_search import LexiconType, LmType, VocabType


ctc_loss_am_args = {
    "state_tying": "monophone-eow",
    "states_per_phone": 1,
    "tdp_scale": 1.0,
    "tdp_transition": (0.0, 0.0, "infinity", 0.0),
    "tdp_silence": (0.0, 0.0, "infinity", 0.0),
    "tdp_nonword": (0.0, 0.0, "infinity", 0.0),
}

ctc_recog_am_args = copy.deepcopy(ctc_loss_am_args)
ctc_recog_am_args.update(
    {
        "state_tying": "monophone",
        "phon_history_length": 0,
        "phon_future_length": 0,
    }
)


def get_ctc_train_step_args(**kwargs) -> Dict:
    default_args = {
        "time_rqmt": 168,
        "mem_rqmt": 24,
        "cpu_rqmt": 6,
        "log_verbosity": 5,
    }
    return recursive_update(default_args, kwargs)


def get_ctc_recog_step_args(num_classes: int, reduction_factor: int = 4, **kwargs) -> Dict:
    default_args = {
        "epochs": ["best"],
        "lm_scales": [0.9],
        "prior_scales": [0.3],
        "use_gpu": False,
        "am_args": copy.deepcopy(ctc_recog_am_args),
        "label_scorer_args": {
            "use_prior": True,
            "num_classes": num_classes,
            "extra_args": {
                "blank_label_index": 0,
                "reduction_factors": reduction_factor,
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
            allow_loop=True,
        ),
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": True,
        },
        "prior_args": {
            "mem_rqmt": 16,
        },
        "rtf": 20,
        "mem": 8,
    }

    return recursive_update(default_args, kwargs)


def get_ctc_flashlight_bpe_recog_step_args(**kwargs) -> Dict:
    default_args = {
        "epochs": ["best"],
        "lm_scales": [2.0],
        "prior_scales": [0.3],
        "feature_type": FeatureType.SAMPLES,
        "lexicon_type": LexiconType.FLASHLIGHT,
        "lm_type": LmType.ARPA_FILE,
        "vocab_type": VocabType.LEXICON_INVENTORY,
        "mem_rqmt": 16,
        "time_rqmt": 24,
    }

    return recursive_update(default_args, kwargs)


def get_ctc_greedy_bpe_recog_step_args(**kwargs) -> Dict:
    default_args = {
        "epochs": ["best"],
        "prior_scales": [0.3],
        "feature_type": FeatureType.SAMPLES,
        "lexicon_type": LexiconType.NONE,
        "lm_type": LmType.NONE,
        "vocab_type": VocabType.LEXICON_INVENTORY,
        "convert_bpe_results": True,
        "mem_rqmt": 16,
        "time_rqmt": 24,
    }

    return recursive_update(default_args, kwargs)


def get_ctc_align_step_args(num_classes: int, reduction_factor: int = 4, **kwargs) -> Dict:
    default_args = {
        "epoch": "best",
        "prior_scale": 0.3,
        "use_gpu": False,
        "am_args": copy.deepcopy(ctc_recog_am_args),
        "alignment_options": {
            "label-pruning": 50,
            "label-pruning-limit": 100000,
        },
        "align_node_options": {
            "allophone-state-graph-builder.topology": "rna",  # No label loop for transducer
        },
        "label_scorer_args": {
            "use_prior": True,
            "num_classes": num_classes,
            "extra_args": {
                "blank_label_index": 0,
                "reduction_factors": reduction_factor,
            },
        },
        "rtf": 5,
        "register_output": False,
    }

    return recursive_update(default_args, kwargs)
