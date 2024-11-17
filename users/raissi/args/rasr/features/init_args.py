__all__ = ["get_feature_extraction_args_8kHz", "get_feature_extraction_args_16kHz"]

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
    mfcc_filter_width = features.filter_width_from_channels(channels=15, f_max=4000)  # = 8000 / 2

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
    mfcc_filter_width = features.filter_width_from_channels(channels=20, f_max=8000)  # = 16000 / 2

    if mfcc_cepstrum_options is None:
        mfcc_cepstrum_options = {
            "normalize": False,
            "outputs": 16,
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
                # "warp_freqbreak": 6600, this is rasr's default
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
        "fb": {
            "filterbank_options": {
                "warping_function": "mel",
                "filter_width": features.filter_width_from_channels(channels=80, warping_function="mel", f_max=8000),
                "normalize": True,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": False,
                },
                "fft_options": None,
                "add_features_output": True,
                "apply_log": True,
                "add_epsilon": True,
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
