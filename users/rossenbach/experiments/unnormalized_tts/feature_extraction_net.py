layernorm_ft_net = {
    "stft": {
        "class": "stft",
        "frame_shift": 200,
        "frame_size": 800,
        "fft_size": 1024,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 1024,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
    },
    "log_mel_features": {
        "class": "layer_norm",
        "from": "log",
    }
}


log10_net = {
    "stft": {
        "class": "stft",
        "frame_shift": 200,
        "frame_size": 800,
        "fft_size": 1024,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 1024,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
    },
    "log10": {
        "from": "log",
        "class": "eval",
        "eval": "source(0) / 2.3026"
    },
    "log_mel_features": {
        "class": "copy",
        "from": "log10",
    }
}

log10_halved_net = {
    "stft": {
        "class": "stft",
        "frame_shift": 200,
        "frame_size": 800,
        "fft_size": 1024,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 1024,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
    },
    "log10": {
        "from": "log",
        "class": "eval",
        "eval": "source(0) / 4.6"
    },
    "log_mel_features": {
        "class": "copy",
        "from": "log10",
    }
}

log10_net_10ms = {
    "stft": {
        "class": "stft",
        "frame_shift": 160,
        "frame_size": 400,
        "fft_size": 512,
        "from": "data:audio_features",
    },
    "abs": {
        "class": "activation",
        "from": "stft",
        "activation": "abs",
    },
    "power": {
        "class": "eval",
        "from": "abs",
        "eval": "source(0) ** 2",
    },
    "mel_filterbank": {
        "class": "mel_filterbank",
        "from": "power",
        "fft_size": 512,
        "nr_of_filters": 80,
        "n_out": 80,
    },
    "log": {
        "from": "mel_filterbank",
        "class": "activation",
        "activation": "safe_log",
        "opts": {"eps": 1e-10},
    },
    "log10": {
        "from": "log",
        "class": "eval",
        "eval": "source(0) / 2.3026"
    },
    "log_mel_features": {
        "class": "copy",
        "from": "log10",
    }
}


def get_roll_augment_net(min_val=0.125, max_val=0.25, broadcast_scale=True):
    log10_net_10ms_aug = {
        "roll": {
            "class": "eval",
            "eval": f"tf.roll(source(0), 1, 0) * tf.random.uniform((1,), {min_val}, {max_val})",
            "from": "data:audio_features",
        },
        "roll_add": {
            "class": "combine",
            "kind": "add",
            "from": ["data:audio_features", "roll"],
        },
        "stft": {
            "class": "stft",
            "frame_shift": 160,
            "frame_size": 400,
            "fft_size": 512,
            "from": "roll_add",
        },
        "abs": {
            "class": "activation",
            "from": "stft",
            "activation": "abs",
        },
        "power": {
            "class": "eval",
            "from": "abs",
            "eval": "source(0) ** 2",
        },
        "mel_filterbank": {
            "class": "mel_filterbank",
            "from": "power",
            "fft_size": 512,
            "nr_of_filters": 80,
            "n_out": 80,
        },
        "log": {
            "from": "mel_filterbank",
            "class": "activation",
            "activation": "safe_log",
            "opts": {"eps": 1e-10},
        },
        "log10": {
            "from": "log",
            "class": "eval",
            "eval": "source(0) / 2.3026"
        },
        "log_mel_features": {
            "class": "copy",
            "from": "log10",
        }
    }
    if broadcast_scale is False:
        log10_net_10ms_aug["roll"]["eval"] = f"tf.roll(source(0), 1, 0) * tf.random.uniform((tf.shape(source(0))[0], 1, 1), {min_val}, {max_val})"

    return log10_net_10ms_aug


def get_roll_augment_net_exponential(min_val=-120, max_val=-12, broadcast_scale=True):
    min_val = min_val/20
    max_val = max_val/20
    log10_net_10ms_aug = {
        "roll": {
            "class": "eval",
            "eval": f"tf.roll(source(0), 1, 0) * (10**tf.random.uniform((tf.shape(source(0))[0], 1, 1), {min_val}, {max_val}))",
            "from": "data:audio_features",
        },
        "roll_add": {
            "class": "combine",
            "kind": "add",
            "from": ["data:audio_features", "roll"],
        },
        "stft": {
            "class": "stft",
            "frame_shift": 160,
            "frame_size": 400,
            "fft_size": 512,
            "from": "roll_add",
        },
        "abs": {
            "class": "activation",
            "from": "stft",
            "activation": "abs",
        },
        "power": {
            "class": "eval",
            "from": "abs",
            "eval": "source(0) ** 2",
        },
        "mel_filterbank": {
            "class": "mel_filterbank",
            "from": "power",
            "fft_size": 512,
            "nr_of_filters": 80,
            "n_out": 80,
        },
        "log": {
            "from": "mel_filterbank",
            "class": "activation",
            "activation": "safe_log",
            "opts": {"eps": 1e-10},
        },
        "log10": {
            "from": "log",
            "class": "eval",
            "eval": "source(0) / 2.3026"
        },
        "log_mel_features": {
            "class": "copy",
            "from": "log10",
        }
    }
    return log10_net_10ms_aug
