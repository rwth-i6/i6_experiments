"""
Definition of all changeable pipeline parameters of the GMM-RASR-Pipeline

The parameters defined here are based on the past experience with training LibriSpeech-100h and 960h models,
but no excessive tuning has been done.
"""
from i6_core.features.filterbank import filter_width_from_channels
from i6_core import cart

from i6_experiments.common.setups.rasr import util

from i6_experiments.common.baselines.librispeech.default_tools import SCTK_BINARY_PATH

from .data import cart_phonemes, cart_steps

def get_init_args():
    dc_detection = False
    samples_options = {
        "audio_format": "wav",
        "dc_detection": dc_detection,
    }

    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),  # loop, forward, skip, exit
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global-and-nonword",
        "nonword_phones": "[LAUGHTER],[NOISE],[VOCALIZEDNOISE]",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            6.0,
        ),  # only used when tying_type = global-and-nonword
    }


    costa_args = {"eval_recordings": True, "eval_lm": True}

    feature_extraction_args = {
        "mfcc": {
            "num_deriv": 2,
            "num_features": None,  # confusing name: number of max features, above number -> clipped
            "mfcc_options": {
                "warping_function": "mel",
                "filter_width": filter_width_from_channels(
                    channels=16, warping_function="mel", f_max=4000
                ),
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": samples_options,
                "cepstrum_options": {
                    "normalize": False,
                    "outputs": 16,  # this is the actual output feature dimension
                    "add_epsilon": not dc_detection,  # when there is no dc-detection we can have log(0) otherwise
                    "epsilon": 1e-10,
                },
                "fft_options": None,
                "add_features_output": True,
            },
        },
        "energy": {
            "energy_options": {
                "without_samples": False,
                "samples_options": samples_options,
                "fft_options": None,
            }
        },
        "gt": {
            "gt_options": {
                "minfreq": 100,
                "maxfreq": 3800,
                "channels": 40,
                "warp_freqbreak": 3700,
                "tempint_type": "hanning",
                "tempint_shift": 0.01,
                "tempint_length": 0.025,
                "flush_before_gap": True,
                "do_specint": False,
                "specint_type": "hanning",
                "specint_shift": 4,
                "specint_length": 9,
                "normalize": True,
                "preemphasis": True,
                "legacy_scaling": False,
                "without_samples": False,
                "samples_options": samples_options,
                "normalization_options": {},
            }
        },
    }

    scorer_args = {
        "sctk_binary_path": SCTK_BINARY_PATH,
    }

    return util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        scorer_args=scorer_args,
        scorer="hub5",
    )


def get_monophone_args():
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
        "feature_flow": "mfcc+deriv+norm",
        "feature_energy_flow_key": "energy,mfcc+deriv+norm",
        "align_iter": 20,
        "splits": 10,
        "accs_per_split": 2,
        "dump_alignment_score_report": True,
    }

    monophone_recognition_args = {
        # GmmSystem.recognition() args:
        "iters": [10],
        "lm_scales": [10],
        "optimize_am_lm_scale": True,
        # meta.System.recog() args:
        "feature_flow": "mfcc+deriv+norm",
        "pronunciation_scales": [1.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam-pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "parallelize_conversion": False,
        "lattice_to_ctm_kwargs": {},
        "rtf": 25,
        "mem": 4,
        "use_gpu": False,
    }

    return util.GmmMonophoneArgs(
        linear_alignment_args, monophone_training_args, monophone_recognition_args
    )


def get_cart_args(
    max_leaves: int = 9001,
    hmm_states: int = 3,
    feature_flow: str = "mfcc+deriv+norm",
):
    """

    :param use_stress_marker: use ARPAbet stress marker, please also check for correct lexicon then
    :param max_leaves:
    :param min_obs:
    :param hmm_states:
    :param feature_flow:
    :param add_unknown:
    :return:
    """

    cart_questions = cart.PythonCartQuestions(
        phonemes=cart_phonemes,
        steps=cart_steps,
        max_leaves=max_leaves,
        hmm_states=hmm_states,
    )

    cart_lda_args = {
        "name": "cart_mono",
        "alignment": "train_mono",
        "initial_flow_key": feature_flow,
        "context_flow_key": feature_flow.split("+")[0],
        "context_size": 9,
        "num_dim": 40,
        "num_iter": 2,
        "eigenvalue_args": {},
        "generalized_eigenvalue_args": {"all": {"verification_tolerance": 1e14}},
    }

    return util.GmmCartArgs(
        cart_questions=cart_questions,
        cart_lda_args=cart_lda_args,
    )


def get_triphone_args():
    triphone_training_args = {
        "name": "tri",
        "initial_alignment": "train_mono",
        "feature_flow": "mfcc+context+lda",
        "splits": 10,
        "accs_per_split": 2,
        "align_extra_rqmt": {"mem": 8},
        "accumulate_extra_rqmt": {"mem": 8},
        "split_extra_rqmt": {"mem": 8},
    }

    triphone_recognition_args = {
        "iters": [8, 10],
        "feature_flow": "mfcc+context+lda",
        "pronunciation_scales": [1.0],
        "lm_scales": [20],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 30,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.tri",
        "alignment": "train_tri",
        "feature_flow_key": "mfcc+context+lda",
    }

    return util.GmmTriphoneArgs(
        training_args=triphone_training_args,
        recognition_args=triphone_recognition_args,
        sdm_args=sdm_args,
    )

def get_cart_reestimation_args(
        max_leaves: int = 9001,
        hmm_states: int = 3,
        feature_flow: str = "mfcc+deriv+norm",
):
    """

    :param use_stress_marker: use ARPAbet stress marker, please also check for correct lexicon then
    :param max_leaves:
    :param min_obs:
    :param hmm_states:
    :param feature_flow:
    :param add_unknown:
    :return:
    """

    cart_questions = cart.PythonCartQuestions(
        phonemes=cart_phonemes,
        steps=cart_steps,
        max_leaves=max_leaves,
        hmm_states=hmm_states,
    )

    cart_lda_args = {
        "name": "cart_tri",
        "alignment": "train_tri",
        "initial_flow_key": feature_flow,
        "context_flow_key": feature_flow.split("+")[0],
        "context_size": 9,
        "num_dim": 40,
        "num_iter": 2,
        "eigenvalue_args": {},
        "generalized_eigenvalue_args": {"all": {"verification_tolerance": 1e14}},
    }

    return util.GmmCartArgs(
        cart_questions=cart_questions,
        cart_lda_args=cart_lda_args,
    )


def get_triphone_second_pass_args():
    triphone_training_args = {
        "name": "tri2",
        "initial_alignment": "train_tri",
        "feature_flow": "mfcc+context+lda",
        "splits": 10,
        "accs_per_split": 2,
        "align_extra_rqmt": {"mem": 8},
        "accumulate_extra_rqmt": {"mem": 8},
        "split_extra_rqmt": {"mem": 8},
    }

    triphone_recognition_args = {
        "iters": [8, 10],
        "feature_flow": "mfcc+context+lda",
        "pronunciation_scales": [1.0],
        "lm_scales": [20],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 30,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.tri2",
        "alignment": "train_tri2",
        "feature_flow_key": "mfcc+context+lda",
    }

    return util.GmmTriphoneArgs(
        training_args=triphone_training_args,
        recognition_args=triphone_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_args():
    vtln_training_args = {
        "feature_flow": {
            "name": "uncached_mfcc+context+lda",
            "lda_matrix_key": "cart_tri",
            "base_flow_key": "uncached_mfcc",
            "context_size": 9,
        },
        "warp_mix": {
            "name": "tri2",
            "alignment": "train_tri2",
            "feature_scorer": "estimate_mixtures_sdm.tri2",
            "splits": 8,
            "accs_per_split": 2,
        },
        "train": {
            "name": "vtln",
            "initial_alignment_key": "train_tri2",
            "splits": 10,
            "accs_per_split": 2,
            "feature_flow": "mfcc+context+lda+vtln",
            "accumulate_extra_rqmt": {"mem": 4},
        },
    }

    vtln_recognition_args = {
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda+vtln",
        "pronunciation_scales": [1.0],
        "lm_scales": [20],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 30,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.vtln",
        "alignment": "train_vtln",
        "feature_flow_key": "mfcc+context+lda+vtln",
    }

    return util.GmmVtlnArgs(
        training_args=vtln_training_args,
        recognition_args=vtln_recognition_args,
        sdm_args=sdm_args,
    )


def get_sat_args():
    sat_training_args = {
        "name": "sat",
        "mixtures": "estimate_mixtures_sdm.tri",
        "alignment": "train_tri2",
        "feature_cache": "mfcc",
        "feature_flow_key": "mfcc+context+lda",
        "cache_regex": "^mfcc.*$",
        "splits": 10,
        "accs_per_split": 2,
    }

    sat_recognition_args = {
        "prev_ctm": util.PrevCtm(
            prev_step_key="tri2",
            pronunciation_scale=1.0,
            lm_scale=20,
            iteration=10,
            optimized_lm=True,
        ),
        "feature_cache": "mfcc",
        "cache_regex": "^mfcc.*$",
        "cmllr_mixtures": "estimate_mixtures_sdm.tri",
        "iters": [8, 10],
        "feature_flow": "mfcc+context+lda",
        "pronunciation_scales": [1.0],
        "lm_scales": [25],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 30,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.sat",
        "alignment": "train_sat",
        "feature_flow_key": "mfcc+context+lda+cmllr",
    }

    return util.GmmSatArgs(
        training_args=sat_training_args,
        recognition_args=sat_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_sat_args():
    vtln_sat_training_args = {
        "name": "vtln+sat",
        "mixtures": "estimate_mixtures_sdm.vtln",
        "alignment": "train_vtln",
        "feature_cache": "mfcc+context+lda+vtln",
        "feature_flow_key": "mfcc+context+lda+vtln",
        "cache_regex": "^.*\\+vtln$",
        "splits": 10,
        "accs_per_split": 2,
    }

    vtln_sat_recognition_args = {
        "prev_ctm": util.PrevCtm(
            prev_step_key="vtln",
            pronunciation_scale=1.0,
            lm_scale=20,
            iteration=10,
            optimized_lm=True,
        ),
        "feature_cache": "mfcc",
        "cache_regex": "^mfcc.*$",
        "cmllr_mixtures": "estimate_mixtures_sdm.vtln",
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda+vtln",
        "pronunciation_scales": [1.0],
        "lm_scales": [25],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 15000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 30,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.vtln+sat",
        "alignment": "train_vtln+sat",
        "feature_flow_key": "mfcc+context+lda+vtln+cmllr",
    }

    return util.GmmVtlnSatArgs(
        training_args=vtln_sat_training_args,
        recognition_args=vtln_sat_recognition_args,
        sdm_args=sdm_args,
    )
