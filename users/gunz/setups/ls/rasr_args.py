__all__ = [
    "get_init_args",
    "get_data_inputs",
]

import copy
from typing import Dict, Optional, Union

# -------------------- Sisyphus --------------------
from sisyphus import tk

# -------------------- Recipes --------------------
import i6_core.features as features
from i6_core.returnn import CompileNativeOpJob

import i6_experiments.common.datasets.librispeech as lbs_dataset
import i6_experiments.common.setups.rasr.util as rasr_util

# -------------------- helpers --------------------
# -------------------- functions --------------------
def get_init_args(
    *,
    dc_detection: bool = False,
    am_extra_args: Optional[Dict] = None,
    mfcc_filter_width: Optional[Union[float, Dict]] = None,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_extra_args: Optional[Dict] = None,
    gt_normalization: bool = True,
    gt_options_extra_args: Optional[Dict] = None,
):
    """
    :param dc_detection:
    :param am_extra_args:
    :param mfcc_filter_width: dict(channels=20, warping_function="mel", f_max=8000, f_min=0) or 268.258
    :param mfcc_cepstrum_options:
    :param mfcc_extra_args:
    :param gt_normalization:
    :param gt_options_extra_args:
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

    return rasr_util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
    )


def get_data_inputs(
    train_corpus="train-other-960",
    add_unknown_phoneme_and_mapping=True,
    use_eval_data_subset: bool = False,
    lm_cfg: dict = None,
):
    corpus_object_dict = lbs_dataset.get_corpus_object_dict(
        audio_format="wav",
        output_prefix="corpora",
    )

    lm = lm_cfg or {
        "filename": lbs_dataset.get_arpa_lm_dict()["4gram"],
        "type": "ARPA",
        "scale": 10,
    }

    train_lexicon = {
        "filename": lbs_dataset.get_g2p_augmented_bliss_lexicon_dict(
            use_stress_marker=False,
            add_unknown_phoneme_and_mapping=add_unknown_phoneme_and_mapping,
        )[train_corpus],
        "normalize_pronunciation": False,
    }

    lexicon = {
        "filename": lbs_dataset.get_bliss_lexicon(),
        "normalize_pronunciation": False,
    }

    train_data_inputs = {}
    dev_data_inputs = {}
    test_data_inputs = {}

    train_data_inputs[train_corpus] = rasr_util.RasrDataInput(
        corpus_object=corpus_object_dict[train_corpus],
        concurrent=300,
        lexicon=train_lexicon,
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
