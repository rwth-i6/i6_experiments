from i6_core.features.filterbank import filter_width_from_channels
from i6_core import cart

from i6_experiments.common.setups.rasr import util
import i6_experiments.common.datasets.tedlium2 as ted_dataset


def get_init_args():
    samples_options = {
        "audio_format": "wav",
        "dc_detection": False,
    }

    am_args = {
        "state_tying": "monophone",
        "states_per_phone": 3,
        "state_repetitions": 1,
        "across_word_model": True,
        "early_recombination": False,
        "tdp_scale": 1.0,
        "tdp_transition": (3.0, 0.0, "infinity", 0.0),
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

    costa_args = {"eval_recordings": True, "eval_lm": False}

    feature_extraction_args = {
        "mfcc": {
            "num_deriv": 2,
            "num_features": None,  # confusing name: number of max features, above number -> clipped
            "mfcc_options": {
                "warping_function": "mel",
                # to be compatible with our old magic number, we have to use 20 features
                "filter_width": filter_width_from_channels(channels=20, warping_function="mel", f_max=8000),
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": samples_options,
                "cepstrum_options": {
                    "normalize": False,
                    "outputs": 16,  # this is the actual output feature dimension
                    "add_epsilon": True,  # when there is no dc-detection we can have log(0) otherwise
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
        "gt": {  # copied from Benedikt October 2023
            "minfreq": 100,
            "maxfreq": 7500,
            "channels": 50,
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
                "dc_detection": False,
            },
            "normalization_options": {},
        },
    }

    return util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_args,
        feature_extraction_args=feature_extraction_args,
        scorer_args=scorer_args,
    )


def get_number_of_segments():
    num_segments = constants.NUM_SEGMENTS["train"]
    for subset in ["clean-360", "other-500"]:
        del num_segments[f"train-{subset}"]
    return num_segments
