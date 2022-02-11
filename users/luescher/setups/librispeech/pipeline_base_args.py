__all__ = [
    "get_init_args",
    "get_monophone_args",
    "get_cart_args",
    "get_triphone_args",
    "get_vtln_args",
    "get_sat_args",
    "get_vtln_sat_args",
    "get_final_output",
    "get_data_inputs",
]

from typing import Dict, Optional, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.cart as cart
import i6_core.rasr as rasr
import i6_core.returnn as returnn

import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.users.luescher.cart.librispeech import FoldedCartQuestions

# -------------------- functions --------------------


def get_init_args(
    *,
    dc_detection: bool = True,
    scorer: Optional[str] = None,
    mfcc_filter_width: Union[float, Dict] = 268.258,
    am_extra_args: Optional[Dict] = None,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_extra_args: Optional[Dict] = None,
    gt_options_extra_args: Optional[Dict] = None,
):
    """
    :param dc_detection:
    :param scorer:
    :param am_extra_args:
    :param mfcc_filter_width: dict(channels=16, warping_function="mel", f_max=8000, f_min=0)
    :param mfcc_cepstrum_options:
    :param mfcc_extra_args:
    :param gt_options_extra_args:
    :return:
    """
    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, 30.0, 0.0),  # loop, forward, skip, exit
        "tdp_silence": (0.0, 3.0, "infinity", 20.0),
        "tying_type": "global",
        "nonword_phones": "",
        "tdp_nonword": (
            0.0,
            3.0,
            "infinity",
            6.0,
        ),  # only used when tying_type = global-and-nonword
    }
    if am_extra_args is not None:
        am_args.update(am_extra_args)

    costa_args = {"eval_recordings": True, "eval_lm": False}
    default_mixture_scorer_args = {"scale": 0.3}

    mfcc_filter_width = (
        features.filter_width_from_channels(**mfcc_filter_width)
        if isinstance(mfcc_filter_width, Dict)
        else mfcc_filter_width
    )

    if mfcc_cepstrum_options is None:
        mfcc_cepstrum_options = {
            "normalize": False,
            "outputs": 16,
            "add_epsilon": False,
        }

    feature_extraction_args = {
        "mfcc": {
            "num_deriv": 2,
            "num_features": None,  # 33 (confusing name: # max features, above -> clipped)
            "mfcc_options": {
                "warping_function": "mel",
                "filter_width": mfcc_filter_width,
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": dc_detection,
                },
                "cepstrum_options": mfcc_cepstrum_options,
                "fft_options": None,
            },
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
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": dc_detection,
                },
                "normalization_options": {},
            }
        },
        "energy": {
            "energy_options": {
                "without_samples": False,
                "samples_options": {"audio_format": "wav", "dc_detection": True},
                "fft_options": {},
            }
        },
    }

    if mfcc_extra_args is not None:
        feature_extraction_args["mfcc"].update(mfcc_extra_args)
    if gt_options_extra_args is not None:
        feature_extraction_args["gt"]["gt_options"].update(gt_options_extra_args)

    return rasr_util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        default_mixture_scorer_args=default_mixture_scorer_args,
        scorer=scorer,
    )


def get_monophone_args(
    allow_zero_weights: bool = False,
    train_align_iter: int = 75,
    feature_flow: str = "mfcc+deriv+norm",
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

        monophone_training_args["align_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        monophone_training_args["accumulate_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        monophone_training_args["split_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        monophone_recognition_args["extra_config"] = allow_zero_weights_extra_config

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
):
    cart_questions_class = FoldedCartQuestions(
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

    return rasr_util.GmmCartArgs(
        cart_questions=cart_questions,
        cart_lda_args=cart_lda_args,
    )


def get_triphone_args(allow_zero_weights: bool = False):
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
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        triphone_training_args["align_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        triphone_training_args["accumulate_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        triphone_training_args["split_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        triphone_recognition_args["extra_config"] = allow_zero_weights_extra_config

    sdm_args = {
        "name": "sdm.tri",
        "alignment": "train_tri",
        "feature_flow_key": "mfcc+context+lda",
    }

    return rasr_util.GmmTriphoneArgs(
        training_args=triphone_training_args,
        recognition_args=triphone_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_args(allow_zero_weights: bool = False):
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
            "accumulate_extra_rqmt": {"mem": 8},
            "align_extra_rqmt": {"mem": 8},
            "split_extra_rqmt": {"mem": 8},
        },
    }

    vtln_recognition_args = {
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda+vtln",
        "pronunciation_scales": [6.0],
        "lm_scales": [21.5, 22.4],
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

        vtln_training_args["train"]["align_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        vtln_training_args["train"]["accumulate_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        vtln_training_args["train"]["split_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        vtln_recognition_args["extra_config"] = allow_zero_weights_extra_config

    sdm_args = {
        "name": "sdm.vtln",
        "alignment": "train_vtln",
        "feature_flow_key": "mfcc+context+lda+vtln",
    }

    return rasr_util.GmmVtlnArgs(
        training_args=vtln_training_args,
        recognition_args=vtln_recognition_args,
        sdm_args=sdm_args,
    )


def get_sat_args(allow_zero_weights: bool = False):
    sat_training_args = {
        "name": "sat",
        "mixtures": "estimate_mixtures_sdm.tri",
        "alignment": "train_tri",
        "feature_cache": "mfcc",
        "feature_flow_key": "mfcc+context+lda",
        "cache_regex": "^mfcc.*$",
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
        "feature_cache": "mfcc",
        "cache_regex": "^mfcc.*$",
        "cmllr_mixtures": "estimate_mixtures_sdm.tri",
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda",
        "pronunciation_scales": [14.0],
        "lm_scales": [14.0],
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

        sat_training_args["align_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        sat_training_args["accumulate_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        sat_training_args["split_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        sat_recognition_args["extra_config"] = allow_zero_weights_extra_config

    sdm_args = {
        "name": "sdm.sat",
        "alignment": "train_sat",
        "feature_flow_key": "mfcc+context+lda+cmllr",
    }

    return rasr_util.GmmSatArgs(
        training_args=sat_training_args,
        recognition_args=sat_recognition_args,
        sdm_args=sdm_args,
    )


def get_vtln_sat_args(allow_zero_weights: bool = False):
    vtln_sat_training_args = {
        "name": "vtln+sat",
        "mixtures": "estimate_mixtures_sdm.vtln",
        "alignment": "train_vtln",
        "feature_cache": "mfcc+context+lda+vtln",
        "feature_flow_key": "mfcc+context+lda+vtln",
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
            21.5,
            10,
            "-optlm",
        ),  # (name, pron_scale, lm_scale, it, opt)
        "feature_cache": "mfcc",
        "cache_regex": "^mfcc.*$",
        "cmllr_mixtures": "estimate_mixtures_sdm.vtln",
        "iters": [8, 10],
        "feature_flow": "uncached_mfcc+context+lda+vtln",
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

        vtln_sat_training_args["align_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        vtln_sat_training_args["accumulate_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        vtln_sat_training_args["split_extra_args"] = {
            "extra_config": allow_zero_weights_extra_config
        }
        vtln_sat_recognition_args["extra_config"] = allow_zero_weights_extra_config

    sdm_args = {
        "name": "sdm.vtln+sat",
        "alignment": "train_vtln+sat",
        "feature_flow_key": "mfcc+context+lda+vtln+cmllr",
    }

    return rasr_util.GmmVtlnSatArgs(
        training_args=vtln_sat_training_args,
        recognition_args=vtln_sat_recognition_args,
        sdm_args=sdm_args,
    )


def get_final_output():
    output_args = rasr_util.OutputArgs("final")

    output_args.define_corpus_type("train-other-960", "train")
    # output_args.define_corpus_type("dev-clean", "dev")
    output_args.define_corpus_type("dev-other", "dev")
    # output_args.define_corpus_type("test-clean", "test")
    # output_args.define_corpus_type("test-other", "test")

    output_args.add_feature_to_extract("gt")

    return output_args


def get_data_inputs(
    train_corpus="train-other-960",
    add_unknown_phoneme_and_mapping=True,
    use_eval_data_subset: bool = False,
):
    corpus_object_dict = lbs_dataset.get_corpus_object_dict(
        audio_format="wav",
        output_prefix="corpora",
    )

    lm = {
        "filename": lbs_dataset.get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    lexicon = {
        "filename": lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=False,
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        )[train_corpus],
        "normalize_pronunciation": False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[train_corpus] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[train_corpus],
        concurrent=300,
        lexicon=lexicon,
    )

    dev_corpus_keys = (
        ["dev-other"] if use_eval_data_subset else ["dev-clean", "dev-other"]
    )
    test_corpus_keys = [] if use_eval_data_subset else ["test-clean", "test-other"]

    for dev_key in dev_corpus_keys:
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=20,
            lexicon=lexicon,
            lm=lm,
        )

    for tst_key in test_corpus_keys:
        test_data_inputs[tst_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[tst_key],
            concurrent=20,
            lexicon=lexicon,
            lm=lm,
        )

    return train_data_inputs, dev_data_inputs, test_data_inputs
