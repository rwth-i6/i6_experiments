from i6_core.features.filterbank import filter_width_from_channels
from i6_core import cart
from i6_core.rasr import RasrConfig

from i6_experiments.common.setups.rasr import gmm_system

from i6_experiments.common.setups.rasr.util import *
from i6_experiments.users.luescher.cart.librispeech import FoldedCartQuestions, UnfoldedCartQuestions


def get_init_args():
    """
    :return:
    :rtype: hybrid.GmmInitArgs
    """
    dc_detection = False
    samples_options = {
        'audio_format': "wav",
        'dc_detection': dc_detection,
    }

    am_args = {
        'state_tying': "monophone",
        'states_per_phone': 3,
        'state_repetitions': 1,
        'across_word_model': True,
        'early_recombination': False,
        'tdp_scale': 1.0,
        'tdp_transition': (3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit
        'tdp_silence': (0.0, 3.0, "infinity", 20.0),
        'tying_type': "global",
        'nonword_phones': "",
        'tdp_nonword': (0.0, 3.0, "infinity", 6.0)  # only used when tying_type = global-and-nonword
    }

    costa_args = {
        'eval_recordings': True,
        'eval_lm': True
    }

    feature_extraction_args = {
        'mfcc': {
            'num_deriv': 2,
            'num_features': None,  # confusing name: number of max features, above number -> clipped
            'mfcc_options': {
                'warping_function': "mel",
                # to be compatible with our old magic number, we have to use 20 features
                'filter_width': filter_width_from_channels(channels=20, warping_function="mel", f_max=8000),
                'normalize': True,
                'normalization_options': None,
                'without_samples': False,
                'samples_options': samples_options,
                'cepstrum_options': {
                    'normalize': False,
                    'outputs': 16, # this is the actual output feature dimension
                    'add_epsilon': not dc_detection, # when there is no dc-detection we can have log(0) otherwise
                    'epsilon': 1e-10,
                },
                'fft_options': None,
            }
        },
        "gt": {
            "gt_options": {
                "minfreq": 100,
                "maxfreq": 7500,
                "channels": 50,
                # "warp_freqbreak": 7400,
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
        'energy': {
            'energy_options': {
                'without_samples': False,
                'samples_options': samples_options,
                'fft_options': {},
            }
        }
    }

    return RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        default_mixture_scorer_args={"scale": 0.3}  # TODO what should this be?
    )


def get_monophone_args():
    linear_alignment_args = {
        'minimum_segment_length': 0,
        'maximum_segment_length': 6000,
        'iterations': 5,
        'penalty': 0,
        'minimum_speech_proportion': .7,
        'save_alignment': False,
        'keep_accumulators': False,
        'extra_merge_args': None,
        'extra_config': None,
        'extra_post_config': None,
    }

    monophone_training_args = {
        'name': 'mono',
        'feature_flow': 'mfcc+deriv+norm',
        'feature_energy_flow_key': 'energy,mfcc+deriv+norm',
        'align_iter': 75,
        'splits': 10,
        'accs_per_split': 2,
        'dump_alignment_score_report': True,
    }

    monophone_recognition_args = {
        # GmmSystem.recognition() args:
        'iters': [10],
        'lm_scales': [10],
        'optimize_am_lm_scale': True,
        # meta.System.recog() args:
        'feature_flow': 'mfcc+deriv+norm',
        'pronunciation_scales': [1.0],
        'lm_lookahead': True,
        'lookahead_options': None,
        'create_lattice': True,
        'eval_single_best': True,
        'eval_best_in_lattice': True,
        'search_parameters': {
            'beam-pruning': 15.0,
            'beam-pruning-limit': 100000,
            'word-end-pruning': 0.5,
            'word-end-pruning-limit': 15000
        },
        'parallelize_conversion': False,
        'lattice_to_ctm_kwargs': {},
        'rtf': 20,
        'mem': 4,
        'use_gpu': False,
    }

    return gmm_system.GmmMonophoneArgs(
        linear_alignment_args,
        monophone_training_args,
        monophone_recognition_args
    )

def get_cart_args(
        folded=True,
        max_leaves: int = 12001,
        min_obs: int = 1000,
        hmm_states: int = 3,
        feature_flow: str = "mfcc+deriv+norm",
        add_unknown: bool = True,
):

    CartQuestions = FoldedCartQuestions if folded else UnfoldedCartQuestions

    cart_questions_class = CartQuestions(
        max_leaves=max_leaves,
        min_obs=min_obs,
        add_unknown=add_unknown,
    )

    cart_questions = cart.PythonCartQuestions(
        phonemes=cart_questions_class.phonemes_boundary_extra,
        steps=cart_questions_class.steps,
        max_leaves=max_leaves,
        hmm_states=hmm_states,
    )

    cart_lda_args = {
        "name": "mono",
        "alignment": "train_mono",
        "initial_flow_key": feature_flow,
        "context_flow_key": feature_flow.split("+")[0],
        "context_size": 9,
        "num_dim": 48,
        "num_iter": 2,
        "eigenvalue_args": {},
        "generalized_eigenvalue_args": {"all": {"verification_tolerance": 1e16}},
    }

    return GmmCartArgs(
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
        "rtf": 20,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.tri",
        "alignment": "train_tri",
        "feature_flow_key": "mfcc+context+lda",
    }

    return GmmTriphoneArgs(
        training_args=triphone_training_args,
        recognition_args=triphone_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_args():
    vtln_training_args = {
        "feature_flow": {
            "name": "uncached_mfcc+context+lda",
            "lda_matrix_key": "mono",
            "base_flow_key": "uncached_mfcc",
            "context_size": 9,
        },
        "warp_mix": {
            "name": "tri",
            "alignment": "train_tri",
            "feature_scorer": "estimate_mixtures_sdm.tri",
            "splits": 8,
            "accs_per_split": 2,
        },
        "train": {
            "name": "vtln",
            "initial_alignment_key": "train_tri",
            "splits": 10,
            "accs_per_split": 2,
            "feature_flow": "mfcc+context+lda+vtln",
        },
    }

    vtln_recognition_args = {
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
        "rtf": 20,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.vtln",
        "alignment": "train_vtln",
        "feature_flow_key": "mfcc+context+lda+vtln",
    }

    return GmmVtlnArgs(
        training_args=vtln_training_args,
        recognition_args=vtln_recognition_args,
        sdm_args=sdm_args,
    )


def get_sat_args():
    sat_training_args = {
        "name": "sat",
        "mixtures": "estimate_mixtures_sdm.tri",
        "alignment": "train_tri",
        "feature_cache": "mfcc",
        "feature_flow_key": "mfcc+context+lda",
        "cache_regex": "^mfcc.*$",
        "splits": 10,
        "accs_per_split": 2,
        "align_keep_values": {7: 20},
    }

    sat_recognition_args = {
        "prev_ctm": PrevCtm(
            prev_step_key="tri",
            pronunciation_scale=1.0,
            lm_scale=25,
            iteration=10,
            optimized_lm=True
        ),
        "feature_cache": "mfcc",
        "cache_regex": "^mfcc.*$",
        "cmllr_mixtures": "estimate_mixtures_sdm.tri",
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda",
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
        "rtf": 20,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.sat",
        "alignment": "train_sat",
        "feature_flow_key": "mfcc+context+lda+cmllr",
    }

    return GmmSatArgs(
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
        "prev_ctm": PrevCtm(
            prev_step_key="vtln",
            pronunciation_scale=1.0,
            lm_scale=25,
            iteration=10,
            optimized_lm=True
        ),
        "feature_cache": "mfcc",
        "cache_regex": "^mfcc.*$",
        "cmllr_mixtures": "estimate_mixtures_sdm.vtln",
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda+vtln",
        "pronunciation_scales": [1.0],
        "lm_scales": [22],
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
        "rtf": 20,
        "mem": 4,
        "parallelize_conversion": True,
    }

    sdm_args = {
        "name": "sdm.vtln+sat",
        "alignment": "train_vtln+sat",
        "feature_flow_key": "mfcc+context+lda+vtln+cmllr",
    }

    return GmmVtlnSatArgs(
        training_args=vtln_sat_training_args,
        recognition_args=vtln_sat_recognition_args,
        sdm_args=sdm_args,
    )
