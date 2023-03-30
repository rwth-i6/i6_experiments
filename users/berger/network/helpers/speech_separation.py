from typing import Dict, List, Tuple, Union
from returnn.tf.util.data import Dim, FeatureDim, SpatialDim
from i6_core.returnn.config import CodeWrapper


def get_speech_separator(
    from_list: str = "data",
    frame_size: int = 512,
    trainable: bool = True,
) -> Tuple[Dict[str, Dict], Dict[str, Dim]]:
    dim_tags = {
        "speaker": FeatureDim("speaker_dim", 2),
        "stft_feature": FeatureDim("stft_output_feature_dim", frame_size // 2 + 1),
        "permutation": FeatureDim("permutation_dim", 2),
    }

    net = {
        "PermutationInvariantTrainingModel_FlattenBatch": {
            "class": "flatten_batch",
            "from": from_list,
            "axis": "T",
            "batch_major": False,
        },
        "dropout_input": {
            "class": "dropout",
            "from": "PermutationInvariantTrainingModel_FlattenBatch",
            "dropout": 0.0,
        },
        "PermutationInvariantTrainingModel_Cast_unnamed_const": {
            "class": "constant",
            "value": 1,
        },
        "PermutationInvariantTrainingModel_Cast": {
            "class": "cast",
            "from": "PermutationInvariantTrainingModel_Cast_unnamed_const",
            "dtype": "float32",
        },
        "PermutationInvariantTrainingModel_add": {
            "class": "combine",
            "from": ["dropout_input", "PermutationInvariantTrainingModel_Cast"],
            "kind": "add",
        },
        "PermutationInvariantTrainingModel_log": {
            "class": "activation",
            "from": "PermutationInvariantTrainingModel_add",
            "activation": "log",
        },
        "blstm_UnflattenBatch": {
            "class": "unflatten_batch",
            "from": "PermutationInvariantTrainingModel_log",
        },
        "blstm": {
            "class": "subnetwork",
            "from": "blstm_UnflattenBatch",
            "subnetwork": {
                "layer0_fwd": {
                    "class": "rec",
                    "from": "data",
                    "unit": "nativelstm2",
                    "n_out": 600,
                    "direction": 1,
                },
                "layer0_bwd": {
                    "class": "rec",
                    "from": "data",
                    "unit": "nativelstm2",
                    "n_out": 600,
                    "direction": -1,
                },
                "layer1_fwd": {
                    "class": "rec",
                    "from": ["layer0_fwd", "layer0_bwd"],
                    "unit": "nativelstm2",
                    "n_out": 600,
                    "direction": 1,
                },
                "layer1_bwd": {
                    "class": "rec",
                    "from": ["layer0_fwd", "layer0_bwd"],
                    "unit": "nativelstm2",
                    "n_out": 600,
                    "direction": -1,
                },
                "layer2_fwd": {
                    "class": "rec",
                    "from": ["layer1_fwd", "layer1_bwd"],
                    "unit": "nativelstm2",
                    "n_out": 600,
                    "direction": 1,
                },
                "layer2_bwd": {
                    "class": "rec",
                    "from": ["layer1_fwd", "layer1_bwd"],
                    "unit": "nativelstm2",
                    "n_out": 600,
                    "direction": -1,
                },
                "output": {
                    "class": "copy",
                    "from": ["layer2_fwd", "layer2_bwd"],
                },
            },
        },
        "blstm_FlattenBatch": {
            "class": "flatten_batch",
            "from": "blstm",
            "axis": "T",
            "batch_major": False,
        },
        "dropout_linear": {
            "class": "dropout",
            "from": "blstm_FlattenBatch",
            "dropout": 0.0,
        },
        "linear1": {
            "class": "linear",
            "from": "dropout_linear",
            "n_out": 1200,
            "with_bias": True,
            "activation": None,
        },
        "relu": {
            "class": "activation",
            "from": "linear1",
            "activation": "relu",
        },
        "linear2": {
            "class": "linear",
            "from": "relu",
            "n_out": frame_size + 2,
            "with_bias": True,
            "activation": None,
        },
        "output_activation": {
            "class": "activation",
            "from": "linear2",
            "activation": "sigmoid",
        },
        "PermutationInvariantTrainingModel_Unflatten": {
            "class": "split_dims",
            "from": "output_activation",
            "axis": "F",
            "dims": (dim_tags["speaker"], dim_tags["stft_feature"]),
        },
        "PermutationInvariantTrainingModel_Transpose": {
            "class": "copy",
            "from": "PermutationInvariantTrainingModel_Unflatten",
        },
        "PermutationInvariantTrainingModel_UnflattenBatch": {
            "class": "unflatten_batch",
            "from": "PermutationInvariantTrainingModel_Transpose",
        },
        "masks": {
            "class": "copy",
            "from": "PermutationInvariantTrainingModel_UnflattenBatch",
        },
        "separated_stft_pit": {
            "class": "subnetwork",
            "from": "masks",
            "subnetwork": {
                "permutation_constant": {
                    "class": "constant",
                    "value": CodeWrapper("np.array([[0, 1], [1, 0]])"),
                    "shape": (dim_tags["permutation"], dim_tags["speaker"]),
                },
                "separated_stft": {
                    "class": "combine",
                    "from": [f"base:{from_list}", "base:masks"],
                    "kind": "mul",
                },
                "separated_stft_permutations": {
                    "class": "gather",
                    "from": "separated_stft",
                    "axis": dim_tags["speaker"],
                    "position": "permutation_constant",
                },
                "permutation_mse": {
                    "class": "subnetwork",
                    "from": "separated_stft_permutations",
                    "subnetwork": {
                        "diff": {
                            "class": "combine",
                            "from": ["data", "base:base:base:target_signals_stft"],
                            "kind": "sub",
                        },
                        "square": {
                            "class": "activation",
                            "from": "diff",
                            "activation": "square",
                        },
                        "output": {
                            "class": "reduce",
                            "from": "square",
                            "mode": "sum",
                            "axes": ["T", "F", dim_tags["speaker"]],
                        },
                    },
                },
                "permutation_argmin": {
                    "class": "reduce",
                    "from": "permutation_mse",
                    "mode": "argmin",
                    "axes": dim_tags["permutation"],
                },
                "permutation_indices": {
                    "class": "gather",
                    "from": "permutation_constant",
                    "position": "permutation_argmin",
                    "axis": dim_tags["permutation"],
                },
                "output": {
                    "class": "gather",
                    "from": "separated_stft",
                    "position": "permutation_indices",
                    "axis": dim_tags["speaker"],
                },
            },
        },
        "output": {
            "class": "copy",
            "from": "separated_stft_pit",
        },
    }

    if not trainable:
        for layer in ["blstm", "linear1", "linear2"]:
            net[layer]["trainable"] = False

    return net, dim_tags


def add_speech_separation(
    network: Dict,
    from_list: Union[str, List[str]] = "data",
    frame_size: int = 512,
    frame_shift: int = 128,
    trainable: bool = True,
) -> Tuple[str, Dict[str, Dim]]:
    sep_net, sep_dim_tags = get_speech_separator(
        frame_size=frame_size, trainable=trainable
    )
    dim_tags = {
        "waveform_time": SpatialDim("waveform_time_dim"),
        "waveform_feature": FeatureDim("waveform_feature_dim", 1),
        "target_time": SpatialDim("target_time_dim"),
        "stft_time": SpatialDim("stft_time_dim"),
    }
    dim_tags.update(sep_dim_tags)

    network.update(
        {
            "stft_complex": {
                "class": "stft",
                "from": from_list,
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "in_spatial_dims": [dim_tags["waveform_time"]],
                "out_spatial_dims": [dim_tags["stft_time"]],
                "out_dim": dim_tags["stft_feature"],
            },
            "stft": {
                "class": "activation",
                "from": "stft_complex",
                "activation": "abs",
            },
            "target_signals_split": {
                "class": "split_dims",
                "from": "data:target_signals",
                "axis": "F",
                "dims": (dim_tags["speaker"], dim_tags["waveform_feature"]),
            },
            "target_signals_stft_complex": {
                "class": "stft",
                "from": "target_signals_split",
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "in_spatial_dims": [dim_tags["waveform_time"]],
                "out_spatial_dims": [dim_tags["stft_time"]],
                "out_dim": dim_tags["stft_feature"],
            },
            "target_signals_stft": {
                "class": "activation",
                "from": "target_signals_stft_complex",
                "activation": "abs",
            },
            "speech_separator": {
                "class": "subnetwork",
                "from": "stft",
                "subnetwork": sep_net,
            },
            "original_phase": {
                "class": "activation",
                "from": "stft_complex",
                "activation": "angle",
                "out_type": {"dtype": "complex64"},
            },
            "separated_with_phase": {
                "class": "eval",
                "from": ["speech_separator", "original_phase"],
                "eval": "tf.cast(source(0), tf.complex64) * tf.math.exp(tf.complex(0., 1.) * tf.cast(source(1), tf.complex64))",
                "out_type": {"dtype": "complex64"},
            },
            "separated_waveforms": {
                "class": "istft",
                "from": "separated_with_phase",
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "in_spatial_dims": [dim_tags["stft_time"]],
                "is_output_layer": True,
            },
            "separated_waveforms_pos": {
                "class": "range_in_axis",
                "from": from_list,
                "axis": "T",
            },
            "separated_waveforms_padded": {
                "class": "pad",
                "from": "separated_waveforms",
                "axes": "T",
                "padding": (0, frame_shift),
            },
            "separated_waveforms_padded_gather": {
                "class": "gather",
                "from": "separated_waveforms_padded",
                "axis": "T",
                "position": "separated_waveforms_pos",
            },
        }
    )

    return "separated_waveforms_padded_gather", dim_tags
