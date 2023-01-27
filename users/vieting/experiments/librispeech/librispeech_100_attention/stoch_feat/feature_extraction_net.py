"""
Contains RETURNN network dicts for feature extraction.
"""
import copy
import numpy as np
from i6_core.returnn.config import CodeWrapper


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


def mel_scale(freq):
  return 1125.0 * np.log(1 + float(freq) / 700)


sampling_rate = 16000
f_min = 0
f_max = sampling_rate / 2.0
fft_size = 512
nr_of_filters = 80
dim_tags = f"""
from returnn.tf.util.data import FeatureDim
fft_bins_dim = FeatureDim("fft_bins", dimension={fft_size // 2 + 1})
center_freqs_range_dim = FeatureDim("center_freqs_range", dimension={nr_of_filters + 2})                                                  
center_freqs_dim = center_freqs_range_dim - 2
"""
dim_tags_batch = f"""
from returnn.tf.util.data import FeatureDim, batch_dim
fft_bins_dim = FeatureDim("fft_bins", dimension={fft_size // 2 + 1})
center_freqs_range_dim = FeatureDim("center_freqs_range", dimension={nr_of_filters + 2})
center_freqs_dim = center_freqs_range_dim - 2
"""
fft_bins_dim = CodeWrapper("fft_bins_dim")
center_freqs_dim = CodeWrapper("center_freqs_dim")
center_freqs_range_dim = CodeWrapper("center_freqs_range_dim")
# fix the following here so the order of the set cannot change in string serialization and break the hash
mel_filterbank_out_shape = CodeWrapper("{fft_bins_dim, center_freqs_dim}")

log10_net_10ms = {
    "log_mel_features": {
        "class": "subnetwork",
        "from": "data:audio_features",
        "subnetwork": {
            "stft": {
                "class": "stft",
                "frame_shift": 160,
                "frame_size": 400,
                "fft_size": fft_size,
                "out_dim": fft_bins_dim,
                "from": "data",
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
            "mel_filterbank_weights": {
                "class": "subnetwork",
                "subnetwork": {
                    "center_freqs": {
                        "class": "subnetwork",
                        "subnetwork": {
                            "linear_range": {
                                "class": "range",
                                "limit": nr_of_filters + 2,
                                "dtype": "float32",
                                "out_spatial_dim": center_freqs_range_dim,
                            },
                            "fmin_fmax": {
                                "class": "eval",
                                "eval": (
                                    f"{mel_scale(f_min)} + "
                                    f"source(0) * {((mel_scale(f_max) - mel_scale(f_min)) / (nr_of_filters + 1))}"),
                                "from": "linear_range"
                            },
                            "inv_mel": {
                                "class": "eval", "eval": "700.0 * (tf.exp(source(0) / 1125) - 1)", "from": "fmin_fmax"},
                            "output": {
                                "class": "eval", "eval": f"source(0) * {fft_size} / {sampling_rate}", "from": "inv_mel"}
                        },
                    },
                    "center_freqs_diff": {
                        "class": "subnetwork",
                        "from": "center_freqs",
                        "subnetwork": {
                            "left": {"class": "slice", "axis": "F", "slice_end": -1, "from": "data"},
                            "right": {"class": "slice", "axis": "F", "slice_start": 1, "from": "data"},
                            "output": {"class": "combine", "kind": "sub", "from": ["right", "left"]},
                        },
                    },
                    "center_freqs_diff_l": {
                        "class": "slice",
                        "axis": "F",
                        "slice_start": 1,
                        "from": "center_freqs_diff",
                        "out_dim": center_freqs_dim,
                    },
                    "center_freqs_diff_r": {
                        "class": "slice",
                        "axis": "F",
                        "slice_end": -1,
                        "from": "center_freqs_diff",
                        "out_dim": center_freqs_dim,
                    },
                    "fft_bins": {
                        "class": "range_in_axis",
                        "axis": "F",
                        "dtype": "float32",
                        "from": "base:power",
                    },
                    "center_freqs_l": {
                        "class": "slice",
                        "axis": "F",
                        "slice_start": 2,
                        "from": "center_freqs",
                        "out_dim": center_freqs_dim,
                    },
                    "center_freqs_r": {
                        "class": "slice",
                        "axis": "F",
                        "slice_end": -2,
                        "from": "center_freqs",
                        "out_dim": center_freqs_dim,
                    },
                    "mel_filterbank_num_l": {
                        "class": "combine",
                        "kind": "sub",
                        "from": ["center_freqs_l", "fft_bins"],
                        "out_shape": mel_filterbank_out_shape,
                    },
                    "mel_filterbank_num_r": {
                        "class": "combine",
                        "kind": "sub",
                        "from": ["fft_bins", "center_freqs_r"],
                        "out_shape": mel_filterbank_out_shape,
                    },
                    "mel_filterbank_l": {
                        "class": "combine", "kind": "truediv",
                        "out_shape": mel_filterbank_out_shape,
                        "from": ["mel_filterbank_num_l", "center_freqs_diff_l"],
                    },
                    "mel_filterbank_r": {
                        "class": "combine", "kind": "truediv",
                        "out_shape": mel_filterbank_out_shape,
                        "from": ["mel_filterbank_num_r", "center_freqs_diff_r"],
                    },
                    "mel_filterbank_lr": {
                        "class": "combine",
                        "kind": "minimum",
                        "from": ["mel_filterbank_l", "mel_filterbank_r"],
                    },
                    "zero": {"class": "constant", "value": 0.},
                    "output": {
                        "class": "combine",
                        "kind": "maximum",
                        "from": ["mel_filterbank_lr", "zero"],
                    },
                },
            },
            "mel_filterbank": {
                "class": "dot",
                "from": ["power", "mel_filterbank_weights"],
                "red1": fft_bins_dim,
                "var1": ["B", "T"],
                "red2": fft_bins_dim,
                "var2": center_freqs_dim,
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
            },
            "output": {
                "class": "copy",
                "from": "log_mel_features",
            },
        },
    },
}

# Version 2 of the base network that does the inverse Mel transformation later to allow applying modifications uniformly
# in Mel scale. Also, fix the previously wrong assignment `_l` and `_r` for left and right boundary of the filters.
log10_net_10ms_v2 = copy.deepcopy(log10_net_10ms)
mel_to_fft_bin_str = f"700.0 * (tf.exp(source(0) / 1125) - 1) * {fft_size} / {sampling_rate}"
log10_net_10ms_v2["log_mel_features"]["subnetwork"]["mel_filterbank_weights"]["subnetwork"] = {
    "fft_bins": {
        "class": "range_in_axis",
        "axis": "F",
        "dtype": "float32",
        "from": "base:power",
    },
    "center_freqs": {
        "class": "subnetwork",
        "subnetwork": {
            "linear_range": {
                "class": "range",
                "limit": nr_of_filters + 2,
                "dtype": "float32",
                "out_spatial_dim": center_freqs_range_dim,
            },
            "output": {
                "class": "eval",
                "eval": (
                    f"{mel_scale(f_min)} + "
                    f"source(0) * {((mel_scale(f_max) - mel_scale(f_min)) / (nr_of_filters + 1))}"),
                "from": "linear_range"
            },
        },
    },
    "center_freqs_l": {
        "class": "slice",
        "axis": "F",
        "slice_end": -2,
        "from": "center_freqs",
        "out_dim": center_freqs_dim,
    },
    "center_freqs_r": {
        "class": "slice",
        "axis": "F",
        "slice_start": 2,
        "from": "center_freqs",
        "out_dim": center_freqs_dim,
    },
    "center_freqs_c": {
        "class": "slice",
        "axis": "F",
        "slice_start": 1,
        "slice_end": -1,
        "from": "center_freqs",
        "out_dim": center_freqs_dim,
    },
    "center_freqs_l_fft": {"class": "eval", "eval": mel_to_fft_bin_str, "from": "center_freqs_l"},
    "center_freqs_r_fft": {"class": "eval", "eval": mel_to_fft_bin_str, "from": "center_freqs_r"},
    "center_freqs_c_fft": {"class": "eval", "eval": mel_to_fft_bin_str, "from": "center_freqs_c"},
    "center_freqs_diff_l": {
        "class": "combine",
        "kind": "sub",
        "from": ["center_freqs_c_fft", "center_freqs_l_fft"],
    },
    "center_freqs_diff_r": {
        "class": "combine",
        "kind": "sub",
        "from": ["center_freqs_r_fft", "center_freqs_c_fft"],
    },
    "mel_filterbank_num": {
        "class": "combine",
        "kind": "sub",
        "from": ["center_freqs_c_fft", "fft_bins"],
        "out_shape": mel_filterbank_out_shape,
    },
    "mel_filterbank_div_l": {
        "class": "combine",
        "kind": "truediv",
        "out_shape": mel_filterbank_out_shape,
        "from": ["mel_filterbank_num", "center_freqs_diff_l"],
    },
    "mel_filterbank_div_r": {
        "class": "combine",
        "kind": "truediv",
        "out_shape": mel_filterbank_out_shape,
        "from": ["mel_filterbank_num", "center_freqs_diff_r"],
    },
    "mel_filterbank_l": {
        "class": "eval",
        "eval": "-source(0) + 1",
        "from": "mel_filterbank_div_l",
    },
    "mel_filterbank_r": {
        "class": "eval",
        "eval": "source(0) + 1",
        "from": "mel_filterbank_div_r",
    },
    "mel_filterbank_lr": {
        "class": "combine",
        "kind": "minimum",
        "from": ["mel_filterbank_l", "mel_filterbank_r"],
    },
    "zero": {"class": "constant", "value": 0.},
    "output": {
        "class": "combine",
        "kind": "maximum",
        "from": ["mel_filterbank_lr", "zero"],
    },
}

pre_emphasis = {
    "class": "subnetwork",
    "from": "data",
    "subnetwork": {
        "shift_0": {"class": "slice", "axis": "T", "slice_end": -1, "from": "data"},
        "shift_0_scale": {"class": "eval", "eval": "source(0) * 1.0", "from": "shift_0"},
        "shift_1": {"class": "slice", "axis": "T", "slice_start": 1, "from": "data"},
        "shift_1_sync": {"class": "reinterpret_data", "from": "shift_1", "size_base": "shift_0_scale"},
        "output": {"class": "combine", "kind": "sub", "from": ["shift_1_sync", "shift_0_scale"]},
    }
}
