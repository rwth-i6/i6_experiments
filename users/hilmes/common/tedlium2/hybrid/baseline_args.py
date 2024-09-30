from i6_core.features import filter_width_from_channels


def get_gammatone_feature_extraction_args():
    return {
        "gt_options": {
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
        }
    }


def get_log_mel_feature_extraction_args():

    return {
        "fb": {
            "filterbank_options": {
                "warping_function": "mel",
                "filter_width": filter_width_from_channels(channels=80, warping_function="mel", f_max=8000),
                "normalize": False,
                "normalization_options": None,
                "without_samples": False,
                "samples_options": {
                    "audio_format": "wav",
                    "dc_detection": False,
                    "scale_input": 0.00003051757
                },
                "fft_options": {"preemphasis": 0.97},
                "add_features_output": True,
                "apply_log": True,
                "add_epsilon": True,
            }
        }
    }

def get_samples_extraction_args():
    return {
        "audio_format": "wav",
        "dc_detection": False,
        "input_options": None,
        "scale_input": 0.00003051757
    }