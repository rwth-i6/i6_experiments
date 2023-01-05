"""
Contains RETURNN network dicts for feature extraction.
"""

log10_net_10ms_ref = {
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
