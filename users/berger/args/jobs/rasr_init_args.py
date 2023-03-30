from typing import Optional, Union, Dict
import i6_core.features as features
import i6_core.cart as cart
import i6_experiments.common.setups.rasr.util as rasr_util


def get_feature_extraction_args_8kHz(
    *,
    dc_detection: bool = False,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_args: Optional[Dict] = None,
    gt_args: Optional[Dict] = None,
) -> Dict:

    mfcc_filter_width = features.filter_width_from_channels(
        channels=15, f_max=4000
    )  # = 8000 / 2

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
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": dc_detection,
                },
                "fft_options": {},
            }
        },
    }

    if mfcc_args is not None:
        feature_extraction_args["mfcc"].update(mfcc_args)
    if gt_args is not None:
        feature_extraction_args["gt"]["gt_options"].update(gt_args)

    return feature_extraction_args


def get_feature_extraction_args_16kHz(
    *,
    dc_detection: bool = False,
    mfcc_cepstrum_options: Optional[Dict] = None,
    mfcc_args: Optional[Dict] = None,
    gt_args: Optional[Dict] = None,
) -> Dict:

    mfcc_filter_width = features.filter_width_from_channels(
        channels=20, f_max=8000
    )  # = 16000 / 2

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
                "warp_freqbreak": 6600,
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
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": dc_detection,
                },
                "fft_options": {},
            }
        },
    }

    if mfcc_args is not None:
        feature_extraction_args["mfcc"].update(mfcc_args)
    if gt_args is not None:
        feature_extraction_args["gt"]["gt_options"].update(gt_args)

    return feature_extraction_args


def get_am_config_args(am_args: Optional[Dict]) -> Dict:
    am_config_args = {
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
    if am_args is not None:
        am_config_args.update(am_args)

    return am_config_args


def get_init_args(
    *,
    sample_rate_kHz: int,
    scorer: Optional[str] = None,
    feature_args: Optional[Dict] = None,
    am_args: Optional[Dict] = None,
    **kwargs,
):
    """
    Initialize acoustic model, feature extraction and scorer arguments
    """
    if feature_args is None:
        feature_args = {}
    if sample_rate_kHz == 8:
        feature_extraction_args = get_feature_extraction_args_8kHz(**feature_args)
    elif sample_rate_kHz == 16:
        feature_extraction_args = get_feature_extraction_args_16kHz(**feature_args)
    else:
        raise NotImplementedError(f"Sample rate {sample_rate_kHz}kHz is not supported")

    am_config_args = get_am_config_args(am_args)

    costa_args = {"eval_recordings": True, "eval_lm": False}

    return rasr_util.RasrInitArgs(
        costa_args=costa_args,
        am_args=am_config_args,
        feature_extraction_args=feature_extraction_args,
        scorer=scorer,
        **kwargs,
    )
