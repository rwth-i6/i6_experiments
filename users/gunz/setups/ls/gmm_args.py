__all__ = [
    "get_monophone_args",
    "get_cart_args",
    "get_triphone_args",
    "get_vtln_args",
    "get_sat_args",
    "get_vtln_sat_args",
    "get_final_output",
]

# -------------------- Sisyphus --------------------
from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.cart as cart
import i6_core.rasr as rasr

import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.common.datasets.librispeech.cart import (
    CartQuestionsWithStress,
    CartQuestionsWithoutStress,
)


# -------------------- helpers --------------------
# -------------------- functions --------------------


def get_monophone_args(
    feature_flow: str = "mfcc+deriv+norm",
    *,
    train_align_iter: int = 75,
    allow_zero_weights: bool = False,
    zero_weights_in: str = "extra_config",
):
    linear_alignment_args = {
        "minimum_segment_length": 0,
        "maximum_segment_length": 6000,
        "iterations": 5,
        "penalty": 0,
        "minimum_speech_proportion": 0.7,
        "save_alignment": False,
        "keep_accumulators": False,
        "extra_merge_args": None,
        "extra_config": None,
        "extra_post_config": None,
    }

    monophone_training_args = {
        "name": "mono",
        "feature_flow": feature_flow,
        "feature_energy_flow_key": f"energy,{feature_flow}",
        "align_iter": train_align_iter,
        "splits": 10,
        "accs_per_split": 2,
    }

    monophone_recognition_args = {
        # GmmSystem.recognition() args:
        "iters": [8, 10],
        "lm_scales": [10.5],
        "optimize_am_lm_scale": True,
        # meta.System.recog() args:
        "feature_flow": feature_flow,
        "pronunciation_scales": [6.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam-pruning": 18.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.75,
            "word-end-pruning-limit": 15000,
        },
        "parallelize_conversion": False,
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "rtf": 50,
        "mem": 8,
        "use_gpu": False,
    }

    monophone_test_recognition_args = None
    # {
    #    "optimize_am_lm_scale": False,
    #    "pronunciation_scales": [1.0],
    #    "lm_scales": [11.0],
    # }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        monophone_training_args["align_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        monophone_training_args["accumulate_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        monophone_training_args["split_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        monophone_recognition_args[zero_weights_in] = allow_zero_weights_extra_config

    sdm_args = {
        "name": "sdm.mono",
        "alignment": "train_mono",
        "feature_flow_key": feature_flow,
    }

    return rasr_util.GmmMonophoneArgs(
        linear_alignment_args=linear_alignment_args,
        training_args=monophone_training_args,
        recognition_args=monophone_recognition_args,
        test_recognition_args=monophone_test_recognition_args,
        sdm_args=sdm_args,
    )


def get_cart_args(
    max_leaves: int = 12001,
    min_obs: int = 1000,
    hmm_states: int = 3,
    feature_flow: str = "mfcc+deriv+norm",
    add_unknown: bool = True,
    name: str = "mono",
    phones: int = 3,
    cart_with_stress: bool = False,
):
    assert phones in [1, 2, 3]

    cart_questions_class = (
        CartQuestionsWithStress(max_leaves=max_leaves, min_obs=min_obs, add_unknown=add_unknown)
        if cart_with_stress
        else CartQuestionsWithoutStress(max_leaves=max_leaves, min_obs=min_obs, add_unknown=add_unknown)
    )

    if phones == 1:
        cart_questions_class.steps = []
    elif phones == 2:
        cart_questions_class.steps[3]["questions"][1]["keys"] = "history[0] central"

    cart_questions = cart.PythonCartQuestions(
        phonemes=cart_questions_class.phonemes_boundary_special,
        steps=cart_questions_class.steps,
        max_leaves=max_leaves,
        hmm_states=hmm_states,
    )

    cart_lda_args = {
        "name": name,
        "alignment": "train_mono",
        "initial_flow_key": feature_flow,
        "context_flow_key": feature_flow.split("+")[0],
        "context_size": 9,
        "num_dim": 48,
        "num_iter": 2,
        "eigenvalue_args": {},
        "generalized_eigenvalue_args": {"all": {"verification_tolerance": 1e16}},
    }

    return rasr_util.GmmCartArgs(
        cart_questions=cart_questions,
        cart_lda_args=cart_lda_args,
    )


def get_triphone_args(
    name: str = "tri",
    initial_alignment: str = "mono",
    feature_flow: str = "mfcc+context+lda",
    allow_zero_weights: bool = False,
    zero_weights_in: str = "extra_config",
):
    triphone_training_args = {
        "name": name,
        "initial_alignment": f"train_{initial_alignment}",
        "feature_flow": feature_flow,
        "splits": 10,
        "accs_per_split": 2,
        "align_extra_rqmt": {"mem": 8},
        "accumulate_extra_rqmt": {"mem": 8},
        "split_extra_rqmt": {"mem": 8},
    }

    triphone_recognition_args = {
        "iters": [8, 10],
        "feature_flow": feature_flow,
        "pronunciation_scales": [6.0],
        "lm_scales": [24.9],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 12.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 50,
        "mem": 12,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        triphone_training_args["align_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        triphone_training_args["accumulate_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        triphone_training_args["split_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        triphone_recognition_args[zero_weights_in] = allow_zero_weights_extra_config

    sdm_args = {
        "name": f"sdm.{name}",
        "alignment": f"train_{name}",
        "feature_flow_key": feature_flow,
    }

    return rasr_util.GmmTriphoneArgs(
        training_args=triphone_training_args,
        recognition_args=triphone_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_args(
    name: str = "vtln",
    feature_flow: str = "mfcc+context+lda",
    initial_alignment_key: str = "tri",
    allow_zero_weights: bool = False,
    zero_weights_in: str = "extra_config",
):
    vtln_training_args = {
        "feature_flow": {
            "name": f"uncached_{feature_flow}",
            "lda_matrix_key": "mono",
            "base_flow_key": f"uncached_{feature_flow.split('+')[0]}",
            "context_size": 9,
        },
        "warp_mix": {
            "name": "tri",
            "alignment": f"train_{initial_alignment_key}",
            "feature_scorer": "estimate_mixtures_sdm.tri",
            "splits": 8,
            "accs_per_split": 2,
        },
        "train": {
            "name": name,
            "initial_alignment_key": f"train_{initial_alignment_key}",
            "splits": 10,
            "accs_per_split": 2,
            "feature_flow": f"{feature_flow}+vtln",
            "accumulate_extra_rqmt": {"mem": 8},
            "align_extra_rqmt": {"mem": 8},
            "split_extra_rqmt": {"mem": 8},
        },
    }

    vtln_recognition_args = {
        "iters": [8, 10],
        "feature_flow": f"uncached_{feature_flow}+vtln",
        "pronunciation_scales": [6.0],
        "lm_scales": [22.4],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 12.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 50,
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        vtln_training_args["train"]["align_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        vtln_training_args["train"]["accumulate_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        vtln_training_args["train"]["split_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        vtln_recognition_args[zero_weights_in] = allow_zero_weights_extra_config

    sdm_args = {
        "name": f"sdm.{name}",
        "alignment": f"train_{name}",
        "feature_flow_key": f"{feature_flow}+vtln",
    }

    return rasr_util.GmmVtlnArgs(
        training_args=vtln_training_args,
        recognition_args=vtln_recognition_args,
        sdm_args=sdm_args,
    )


def get_sat_args(
    name: str = "sat",
    feature_flow: str = "mfcc+context+lda",
    initial_mixture: str = "estimate_mixtures_sdm.tri",
    initial_alignment: str = "tri",
    allow_zero_weights: bool = False,
    zero_weights_in: str = "extra_config",
):
    feature_base_cache = feature_flow.split("+")[0]
    sat_training_args = {
        "name": name,
        "mixtures": initial_mixture,
        "alignment": f"train_{initial_alignment}",
        "feature_cache": feature_base_cache,
        "feature_flow_key": feature_flow,
        "cache_regex": f"^{feature_base_cache}.*$",
        "splits": 10,
        "accs_per_split": 2,
        "align_keep_values": {7: tk.gs.JOB_DEFAULT_KEEP_VALUE},
        "accumulate_extra_rqmt": {"mem": 8},
        "align_extra_rqmt": {"mem": 8},
        "split_extra_rqmt": {"mem": 8},
    }

    sat_recognition_args = {
        "prev_ctm": (
            "tri",
            6.0,
            24.9,
            10,
            "-optlm",
        ),  # (name, pron_scale, lm_scale, it, opt)
        "feature_cache": feature_base_cache,
        "cache_regex": f"^{feature_base_cache}.*$",
        "cmllr_mixtures": initial_mixture,
        "iters": [8, 10],
        "feature_flow": f"uncached_{feature_flow}",
        "pronunciation_scales": [6.0],
        "lm_scales": [30.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 12.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 50,
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        sat_training_args["align_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        sat_training_args["accumulate_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        sat_training_args["split_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        sat_recognition_args[zero_weights_in] = allow_zero_weights_extra_config

    sdm_args = {
        "name": f"sdm.{name}",
        "alignment": f"train_{name}",
        "feature_flow_key": f"{feature_flow}+cmllr",
    }

    return rasr_util.GmmSatArgs(
        training_args=sat_training_args,
        recognition_args=sat_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_sat_args(
    name: str = "vtln+sat",
    feature_flow: str = "mfcc+context+lda+vtln",
    initial_mixture: str = "estimate_mixtures_sdm.vtln",
    initial_alignment: str = "vtln",
    allow_zero_weights: bool = False,
    zero_weights_in: str = "extra_config",
):
    feature_base_cache = feature_flow.split("+")[0]
    vtln_sat_training_args = {
        "name": name,
        "mixtures": initial_mixture,
        "alignment": f"train_{initial_alignment}",
        "feature_cache": feature_flow,
        "feature_flow_key": feature_flow,
        "cache_regex": "^.*\\+vtln$",
        "splits": 10,
        "accs_per_split": 2,
        "accumulate_extra_rqmt": {"mem": 8},
        "align_extra_rqmt": {"mem": 8},
        "split_extra_rqmt": {"mem": 8},
    }

    vtln_sat_recognition_args = {
        "prev_ctm": (
            "vtln",
            6.0,
            22.4,
            10,
            "-optlm",
        ),  # (name, pron_scale, lm_scale, it, opt)
        "feature_cache": feature_base_cache,
        "cache_regex": f"^{feature_base_cache}.*$",
        "cmllr_mixtures": initial_mixture,
        "iters": [8, 10],
        "feature_flow": f"uncached_{feature_flow}",
        "pronunciation_scales": [6.0],
        "lm_scales": [30.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 12.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 50,
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        vtln_sat_training_args["align_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        vtln_sat_training_args["accumulate_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        vtln_sat_training_args["split_extra_args"] = {zero_weights_in: allow_zero_weights_extra_config}
        vtln_sat_recognition_args[zero_weights_in] = allow_zero_weights_extra_config

    sdm_args = {
        "name": f"sdm.{name}",
        "alignment": f"train_{name}",
        "feature_flow_key": f"{feature_flow}+cmllr",
    }

    return rasr_util.GmmVtlnSatArgs(
        training_args=vtln_sat_training_args,
        recognition_args=vtln_sat_recognition_args,
        sdm_args=sdm_args,
    )


def get_final_output(name="final"):
    output_args = rasr_util.OutputArgs(name)

    output_args.define_corpus_type("train-other-960", "train")
    output_args.define_corpus_type("dev-clean", "dev")
    output_args.define_corpus_type("dev-other", "dev")
    output_args.define_corpus_type("test-clean", "test")
    output_args.define_corpus_type("test-other", "test")

    output_args.add_feature_to_extract("gt")

    return output_args
