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

from typing import Any, Dict, Optional, Tuple, Union

# -------------------- Sisyphus --------------------

from sisyphus import tk

# -------------------- Recipes --------------------

import i6_core.features as features
import i6_core.cart as cart
import i6_core.corpus as corpus_recipe
import i6_core.rasr as rasr

import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.common.setups.rasr.util as rasr_util
from i6_experiments.common.baselines.librispeech.default_tools import SCTK_BINARY_PATH
from i6_experiments.common.datasets.librispeech.cart import (
    CartQuestionsWithoutStress,
    CartQuestionsWithStress,
)
from i6_experiments.common.helpers.g2p import G2PBasedOovAugmenter

# -------------------- helpers --------------------
# -------------------- functions --------------------


def get_init_args(
    *,
    dc_detection: bool = False,
    scorer: Optional[str] = None,
    scorer_args: Optional[Dict[str, Any]] = None,
    am_extra_args: Optional[Dict] = None,
    mfcc_filter_width: Optional[Union[float, Dict]] = None,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_extra_args: Optional[Dict] = None,
    gt_normalization: bool = True,
    gt_options_extra_args: Optional[Dict] = None,
    tying_type: str = "global",
    nonword_phones: str = "",
    tdp_transition: Tuple[
        Union[float, str], Union[float, str], Union[float, str], Union[float, str]
    ] = (3.0, 0.0, "infinity", 0.0),
    tdp_silence: Tuple[
        Union[float, str], Union[float, str], Union[float, str], Union[float, str]
    ] = (0.0, 3.0, "infinity", 20.0),
    tdp_nonword: Tuple[
        Union[float, str], Union[float, str], Union[float, str], Union[float, str]
    ] = (
        0.0,
        3.0,
        "infinity",
        6.0,
    ),
):
    """
    :param dc_detection:
    :param scorer:
    :param scorer_args:
    :param am_extra_args:
    :param mfcc_filter_width: dict(channels=20, warping_function="mel", f_max=8000, f_min=0) or 268.258
    :param mfcc_cepstrum_options:
    :param mfcc_extra_args:
    :param gt_normalization:
    :param gt_options_extra_args:
    :param tying_type:
    :param nonword_phones:
    :param tdp_transition:
    :param tdp_silence:
    :param tdp_nonword:
    :return:
    """
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
        "tying_type": tying_type,
        "nonword_phones": nonword_phones,
        "tdp_transition": tdp_transition,  # loop, forward, skip, exit
        "tdp_silence": tdp_silence,  # loop, forward, skip, exit
        "tdp_nonword": tdp_nonword,  # only used when tying_type = global-and-nonword
    }
    if am_extra_args is not None:
        am_args.update(am_extra_args)

    costa_args = {"eval_recordings": True, "eval_lm": True}

    if mfcc_filter_width is None:
        mfcc_filter_width = {
            "channels": 20,
            "warping_function": "mel",
            "f_max": 8000,
            "f_min": 0,
        }

    mfcc_filter_width = (
        features.filter_width_from_channels(**mfcc_filter_width)
        if isinstance(mfcc_filter_width, Dict)
        else mfcc_filter_width
    )

    if mfcc_cepstrum_options is None:
        mfcc_cepstrum_options = {
            "normalize": False,
            "outputs": 16,  # this is the actual output feature dimension
            "add_epsilon": not dc_detection,  # when there is no dc-detection we can have log(0) otherwise
            "epsilon": 1e-10,
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
                "samples_options": samples_options,
                "cepstrum_options": mfcc_cepstrum_options,
                "fft_options": None,
                "add_features_output": True,
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
                "normalize": gt_normalization,
                "preemphasis": True,
                "legacy_scaling": False,
                "without_samples": False,
                "samples_options": samples_options,
                "normalization_options": {},
            }
        },
        "energy": {
            "energy_options": {
                "without_samples": False,
                "samples_options": samples_options,
                "fft_options": {},
            }
        },
    }

    if mfcc_extra_args is not None:
        feature_extraction_args["mfcc"].update(mfcc_extra_args)
    if gt_options_extra_args is not None:
        feature_extraction_args["gt"]["gt_options"].update(gt_options_extra_args)

    if scorer_args is None:
        scorer_args = {"sctk_binary_path": SCTK_BINARY_PATH}
    elif "sctk_binary_path" not in scorer_args.keys():
        scorer_args.update({"sctk_binary_path": SCTK_BINARY_PATH})

    return rasr_util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        scorer=scorer,
        scorer_args=scorer_args,
    )


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
        "dump_alignment_score_report": True,
    }

    monophone_recognition_args = {
        # GmmSystem.recognition() args:
        "iters": [8, 10],
        "lm_scales": [10.5],
        "optimize_am_lm_scale": True,
        # meta.System.recog() args:
        "feature_flow": feature_flow,
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
            "word-end-pruning-limit": 10000,
        },
        "parallelize_conversion": False,
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "rtf": 20,
        "mem": 4,
        "use_gpu": False,
    }

    monophone_test_recognition_args = None

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        monophone_training_args["align_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        monophone_training_args["accumulate_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        monophone_training_args["split_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
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
    use_stress_marker: bool = False,
    max_leaves: int = 12001,
    min_obs: int = 1000,
    hmm_states: int = 3,
    feature_flow: str = "mfcc+deriv+norm",
    add_unknown: bool = False,
):
    CartQuestions = (
        CartQuestionsWithStress if use_stress_marker else CartQuestionsWithoutStress
    )
    cart_questions_class = CartQuestions(
        max_leaves=max_leaves,
        min_obs=min_obs,
        add_unknown=add_unknown,
    )

    cart_questions = cart.PythonCartQuestions(
        phonemes=cart_questions_class.phonemes_boundary_special,
        steps=cart_questions_class.steps,
        max_leaves=max_leaves,
        hmm_states=hmm_states,
    )

    cart_lda_args = {
        "name": "cart_mono",
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
        "pronunciation_scales": [1.0],
        "lm_scales": [25.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 20,
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        triphone_training_args["align_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        triphone_training_args["accumulate_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        triphone_training_args["split_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
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
            "lda_matrix_key": "cart_mono",
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
        "pronunciation_scales": [1.0],
        "lm_scales": [25.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 20,
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        vtln_training_args["train"]["align_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        vtln_training_args["train"]["accumulate_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        vtln_training_args["train"]["split_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
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
        "accumulate_extra_rqmt": {"mem": 8},
        "align_extra_rqmt": {"mem": 8},
        "split_extra_rqmt": {"mem": 8},
    }

    sat_recognition_args = {
        "prev_ctm": rasr_util.PrevCtm(
            prev_step_key="tri",
            pronunciation_scale=1.0,
            lm_scale=25.0,
            iteration=10,
            optimized_lm=True,
        ),
        "feature_cache": feature_base_cache,
        "cache_regex": f"^{feature_base_cache}.*$",
        "cmllr_mixtures": initial_mixture,
        "iters": [8, 10],
        "feature_flow": f"uncached_{feature_flow}",
        "pronunciation_scales": [1.0],
        "lm_scales": [25.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
        },
        "lattice_to_ctm_kwargs": {
            "fill_empty_segments": False,
            "best_path_algo": "bellman-ford",
        },
        "optimize_am_lm_scale": True,
        "rtf": 20,
        "mem": 8,
        "parallelize_conversion": True,
    }

    if allow_zero_weights:
        allow_zero_weights_extra_config = rasr.RasrConfig()
        allow_zero_weights_extra_config.allow_zero_weights = True

        sat_training_args["align_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        sat_training_args["accumulate_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        sat_training_args["split_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
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
        "prev_ctm": rasr_util.PrevCtm(
            prev_step_key="vtln",
            pronunciation_scale=1.0,
            lm_scale=25.0,
            iteration=10,
            optimized_lm=True,
        ),
        "feature_cache": feature_base_cache,
        "cache_regex": f"^{feature_base_cache}.*$",
        "cmllr_mixtures": initial_mixture,
        "iters": [8, 10],
        "feature_flow": f"uncached_{feature_flow}",
        "pronunciation_scales": [1.0],
        "lm_scales": [25.0],
        "lm_lookahead": True,
        "lookahead_options": None,
        "create_lattice": True,
        "eval_single_best": True,
        "eval_best_in_lattice": True,
        "search_parameters": {
            "beam_pruning": 15.0,
            "beam-pruning-limit": 100000,
            "word-end-pruning": 0.5,
            "word-end-pruning-limit": 10000,
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
            zero_weights_in: allow_zero_weights_extra_config
        }
        vtln_sat_training_args["accumulate_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
        vtln_sat_training_args["split_extra_args"] = {
            zero_weights_in: allow_zero_weights_extra_config
        }
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


def get_align_dev_args() -> rasr_util.ForcedAlignmentArgs:
    use_stress_marker = False
    use_g2p_training = True
    alias_path = "g2p_forced_alignment"

    kernel_lexicon = lbs_dataset.get_bliss_lexicon(
        use_stress_marker=use_stress_marker,
        add_unknown_phoneme_and_mapping=not use_g2p_training,
    )

    dev_clean_other_corpus = corpus_recipe.MergeCorporaJob(
        [
            lbs_dataset.get_bliss_corpus_dict("wav", output_prefix=alias_path)[
                "dev-clean"
            ],
            lbs_dataset.get_bliss_corpus_dict("wav", output_prefix=alias_path)[
                "dev-other"
            ],
        ],
        name="dev-clean-other",
        merge_strategy=corpus_recipe.MergeStrategy.FLAT,
    ).out_merged_corpus

    g2p_augmenter = G2PBasedOovAugmenter(
        original_bliss_lexicon=kernel_lexicon,
        train_lexicon=kernel_lexicon,
    )
    forced_align_lexicon = g2p_augmenter.get_g2p_augmented_bliss_lexicon(
        bliss_corpus=dev_clean_other_corpus,
        corpus_name="dev-clean-other",
        alias_path=alias_path,
    )

    return rasr_util.ForcedAlignmentArgs(
        name="align_dev-clean-other",
        target_corpus_keys=["dev-clean", "dev-other"],
        flow="uncached_mfcc+context+lda+vtln+cmllr",
        feature_scorer="train_vtln+sat",
        bliss_lexicon={
            "filename": forced_align_lexicon,
            "normalize_pronunciation": False,
        },
        rtf=5.0,
    )


def get_final_output():
    output_args = rasr_util.OutputArgs("final")

    output_args.define_corpus_type("train-other-960", "train")
    # output_args.define_corpus_type("dev-clean", "dev")
    output_args.define_corpus_type("dev-other", "dev")
    output_args.define_corpus_type("dev-clean_forced-align", "dev")
    output_args.define_corpus_type("dev-other_forced-align", "dev")
    # output_args.define_corpus_type("test-clean", "test")
    # output_args.define_corpus_type("test-other", "test")

    output_args.add_feature_to_extract("gt")

    return output_args


def get_data_inputs(
    train_corpus="train-other-960",
    add_unknown_phoneme_and_mapping: bool = False,
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

    use_stress_marker = False

    original_bliss_lexicon = {
        "filename": lbs_dataset.get_bliss_lexicon(
            use_stress_marker=use_stress_marker,
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        ),
        "normalize_pronunciation": False,
    }

    augmented_bliss_lexicon = {
        "filename": lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=use_stress_marker,
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
        lexicon=augmented_bliss_lexicon,
    )

    dev_corpus_keys = (
        ["dev-other"] if use_eval_data_subset else ["dev-clean", "dev-other"]
    )
    test_corpus_keys = [] if use_eval_data_subset else ["test-clean", "test-other"]

    for dev_key in dev_corpus_keys:
        dev_data_inputs[dev_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[dev_key],
            concurrent=20,
            lexicon=original_bliss_lexicon,
            lm=lm,
        )

    for tst_key in test_corpus_keys:
        test_data_inputs[tst_key] = rasr_util.RasrDataInput(
            corpus_object=corpus_object_dict[tst_key],
            concurrent=20,
            lexicon=original_bliss_lexicon,
            lm=lm,
        )

    return train_data_inputs, dev_data_inputs, test_data_inputs
