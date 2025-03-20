import os.path
from typing import Dict, List, Optional, Tuple, Union, Any

from sisyphus import tk
from sisyphus.delayed_ops import DelayedFunction
import i6_core.rasr as rasr
from i6_core.am.config import acoustic_model_config
from i6_core.returnn import CodeWrapper
from i6_experiments.users.berger.network.helpers.conformer import add_conformer_stack as add_conformer_stack_simon
from ....ctc.feat.network_helpers.specaug import add_specaug_layer, add_specaug_layer_v2
from ....ctc.feat.network_helpers.specaug_sort_layer2 import add_specaug_layer as add_specaug_layer_sort_layer2
from ....ctc.feat.network_helpers.specaug_stft import add_specaug_layer as add_specaug_layer_stft
from ....ctc.feat.network_helpers.conformer_wei import add_conformer_stack as add_conformer_stack_wei
from ....ctc.feat.network_helpers.conformer_wei import add_vgg_stack as add_vgg_stack_wei


def segmental_loss(source):
    """
    L_boost of speech segment
    """
    import tensorflow as tf

    loss = source(0, enforce_batch_major=True)
    mask = source(1, enforce_batch_major=True)
    return tf.where(mask, loss, tf.zeros_like(loss))


def rnnt_loss_compressed(sources, blank_label=0):
    from returnn.extern_private.BergerMonotonicRNNT import rnnt_loss

    logits = sources(0, as_data=True, auto_convert=False)
    targets = sources(1, as_data=True, auto_convert=False)
    encoder = sources(2, as_data=True, auto_convert=False)

    loss = rnnt_loss(
        logits.placeholder,
        targets.get_placeholder_as_batch_major(),
        encoder.get_sequence_lengths(),
        targets.get_sequence_lengths(),
        blank_label=blank_label,
        input_type="logit",
    )
    loss.set_shape((None,))
    return loss


def subtract_ilm(transducer_prob, lm_prob, scale=0.3):
    """
    Decoding only (no time dim) + renormalize out lm index 0
    """
    import tensorflow as tf

    sb = tf.expand_dims(lm_prob[:, 0], -1)  # (B, 1)
    lm_label = lm_prob[:, 1:]  # (B, F)
    norm = tf.pow(lm_label / (1.0 - sb), scale)
    norm = tf.concat([tf.ones(tf.shape(sb)), norm], axis=-1)
    return transducer_prob / norm


def add_transducer_viterbi_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    l2: Optional[float] = None,
    dropout: Optional[float] = None,
    recognition: bool = False,
    **kwargs,
):
    network[name] = {
        "class": "rec",
        "from": from_list,
        "target": "classes",
        "cheating": False,
        "unit": {
            "embedding": {
                "L2": l2,
                "activation": None,
                "class": "linear",
                "from": "base:mask_label",
                "n_out": 128,
                "with_bias": False,
            },
            "mask_embedding": {
                "axes": "T",
                "class": "pad",
                "from": "embedding",
                "mode": "constant",
                "padding": (1, 0),
                "value": 0,
            },
            "label_lm_1": {
                "L2": l2,
                "activation": "tanh",
                "class": "linear",
                "dropout": dropout,
                "from": ["embedding" if recognition else "mask_embedding"],
                "n_out": 640,
            },
            "label_lm_2": {
                "L2": l2,
                "activation": "tanh",
                "class": "linear",
                "dropout": dropout,
                "from": "label_lm_1",
                "n_out": 640,
            },
            "mask_flag": {
                "amount": 1,
                "axis": "T",
                "class": "shift_axis",
                "from": "base:mask_flag",
                "pad": True,
            },
            "unmask_context": {
                "class": "unmask",
                "from": "label_lm_2",
                "mask": "mask_flag",
                "skip_initial": True,
            },
            "unmask_context_reinterpret": {
                "class": "reinterpret_data",
                "from": "unmask_context",
                "size_base": "data:classes",
            },
            "joint_encoding": {
                "L2": l2,
                "activation": "tanh",
                "class": "linear",
                "dropout": dropout,
                "from": ["data:source", "label_lm_2" if recognition else "unmask_context_reinterpret"],
                "n_out": 1024,
            },
            "ce_loss": {
                "class": "loss",
                "from": "output",
                "loss_": "ce",
            },
            "segmental_loss": {
                "class": "eval",
                "eval": "self.network.get_config().typed_value('segmental_loss')(source)",
                "from": ["ce_loss", "base:mask_flag"],
                "loss": "as_is",
                "loss_opts": {"scale": 5.0},
            },
            "output": {
                "class": "softmax",
                "from": "joint_encoding",
                "loss": "ce",
                "loss_opts": {"label_smoothing": 0.2},
            },
        },
    }
    if recognition:
        network[name]["unit"].update({
            "label_context": {
                "class": "choice",
                "from": "output",
                "input_type": "log_prob",
                "target": "classes",
                "beam_size": 1,
                "initial_output": num_outputs + 1,
            },
            "embedding": {
                "L2": l2,
                "activation": None,
                "class": "linear",
                "from": "prev:label_context",
                "n_out": 128,
                "with_bias": False,
                "initial_output": None,
                "safe_embedding": True,
            },
            "output": {
                "class": "linear",
                "from": "joint_encoding",
                "activation": "log_softmax",
                "n_out": num_outputs,
            },
        })
        for layer in [
            "ce_loss", "mask_embedding", "mask_flag", "segmental_loss", "unmask_context", "unmask_context_reinterpret"
        ]:
            network[name]["unit"].pop(layer)
    return name


def add_transducer_fullsum_output_layer(
    network: Dict,
    from_list: Union[str, List[str]],
    num_outputs: int,
    name: str = "output",
    l2: Optional[float] = None,
    dropout: Optional[float] = None,
    recognition: bool = False,
    **kwargs,
):
    if recognition:
        network[name] = {
            "class": "rec",
            "cheating": False,
            "from": from_list,
            "unit": {
                "label_context": {
                    "class": "choice",
                    "from": "output_norm_ilm",
                    "target": "classes",
                    "beam_size": 1,
                    "cheating": False,
                    "initial_output": num_outputs + 1,
                },
                "embedding": {
                    "L2": l2,
                    "activation": None,
                    "class": "linear",
                    "from": "prev:label_context",
                    "n_out": 128,
                    "with_bias": False,
                    "initial_output": None,
                    "safe_embedding": True,
                },
                "label_lm_1": {
                    "L2": l2,
                    "activation": "tanh",
                    "class": "linear",
                    "dropout": 0.0,
                    "from": "embedding",
                    "n_out": 640,
                },
                "label_lm_2": {
                    "L2": l2,
                    "activation": "tanh",
                    "class": "linear",
                    "dropout": 0.0,
                    "from": "label_lm_1",
                    "n_out": 640,
                },
                "zero_encoder": {
                    "class": "pad",
                    "axes": "F",
                    "from": "label_lm_2",
                    "mode": "constant",
                    "padding": (512, 0),
                    "value": 0,
                },
                "joint_encoding": {
                    "L2": l2,
                    "activation": "tanh",
                    "class": "linear",
                    "dropout": 0.0,
                    "from": ["data:source", "label_lm_2"],
                    "n_out": 1024,
                },
                "output": {
                    "class": "softmax",
                    "from": "joint_encoding",
                    "n_out": num_outputs,
                },
                "ilm_joint": {
                    "L2": l2,
                    "activation": "tanh",
                    "class": "linear",
                    "dropout": 0.0,
                    "from": "zero_encoder",
                    "n_out": 1024,
                    "reuse_params": "joint_encoding",
                },
                "ilm_output": {
                    "class": "softmax",
                    "from": "ilm_joint",
                    "is_output_layer": True,
                    "n_out": num_outputs,
                    "reuse_params": "output",
                },
                "output_norm_ilm": {
                    "class": "eval",
                    "from": ["output", "ilm_output"],
                    "n_out": num_outputs,
                    "eval": (
                        "self.network.get_config().typed_value('subtract_ilm')("
                        "source(0, auto_convert=False, enforce_batch_major=True), "
                        "source(1, auto_convert=False, enforce_batch_major=True), "
                        "scale=0.100000)"
                    ),
                },
            },
        }
    else:
        network[name] = {
            "class": "subnetwork",
            "from": from_list,
            "subnetwork": {
                "rec": {
                    "class": "subnetwork",
                    "from": "data",
                    "subnetwork": {
                        "embedding": {
                            "L2": l2,
                            "activation": None,
                            "class": "linear",
                            "from": "base:base:mask_label",
                            "n_out": 128,
                            "with_bias": False,
                        },
                        "mask_embedding": {
                            "axes": "T",
                            "class": "pad",
                            "from": "embedding",
                            "mode": "constant",
                            "padding": (1, 0),
                            "value": 0,
                        },
                        "label_lm_1": {
                            "L2": l2,
                            "activation": "tanh",
                            "class": "linear",
                            "dropout": dropout,
                            "from": ["embedding" if recognition else "mask_embedding"],
                            "n_out": 640,
                        },
                        "label_lm_2": {
                            "L2": l2,
                            "activation": "tanh",
                            "class": "linear",
                            "dropout": dropout,
                            "from": "label_lm_1",
                            "n_out": 640,
                        },
                        "compress_concat": {
                            "class": "compressed_concat",
                            "from": ["base:base:encoder", "label_lm_2"],
                        },
                        "joint_encoding": {
                            "L2": l2,
                            "activation": "tanh",
                            "class": "linear",
                            "dropout": dropout,
                            "from": ["compress_concat"],
                            "n_out": 1024,
                            "out_type": {
                                "batch_dim_axis": None,
                                "shape": (None, 1024),
                                "time_dim_axis": 0,
                            },
                        },
                        "mask_label_int32": {
                            "class": "cast",
                            "dtype": "int32",
                            "from": "base:base:mask_label",
                        },
                        "rnnt_loss": {
                            "class": "eval",
                            "from": [
                                "output",
                                "mask_label_int32",
                                "base:base:encoder",
                            ],
                            "loss": "as_is",
                            "out_type": {
                                "batch_dim_axis": 0,
                                "time_dim_axis": None,
                                "shape": (),
                                "dim": None,
                            },
                            "eval": "self.network.get_config().typed_value('rnnt_loss_compressed')(source, blank_label=0)",
                        },
                        "output": {
                            "class": "linear",
                            "from": "joint_encoding",
                            "activation": None,
                            "n_out": 88,
                            "out_type": {
                                "batch_dim_axis": None,
                                "shape": (None, 88),
                                "time_dim_axis": 0,
                            },
                        },
                    },
                },
                "output": {"class": "copy", "from": "rec"},
            },
        }

    return name


def add_transducer_mbr_layers(
    network: Dict,
    from_list: Union[str, List[str]],
):
    network.update({
        "nbest_classes_size_dense": {
            "class": "reinterpret_data",
            "from": "data:nbest_classes_size",
            "set_sparse": False,
        },
        "nbest_classes_size_int32": {
            "class": "cast",
            "dtype": "int32",
            "from": "nbest_classes_size_dense",
        },
        "nbest_risk_dense": {
            "class": "reinterpret_data",
            "from": "data:nbest_risk",
            "set_sparse": False,
        },
        "nbest_risk_float32": {
            "class": "cast",
            "dtype": "float32",
            "from": "nbest_risk_dense",
        },
        "nbest_score_float32": {
            "class": "cast",
            "dtype": "float32",
            "from": "data:nbest_score",
        },
        "nbest_classes_int32": {
            "class": "cast",
            "dtype": "int32",
            "from": "data:nbest_classes",
        },
        "nbest_classes_sparse": {
            "class": "reinterpret_data",
            "set_sparse": True,
            "set_sparse_dim": 88,
            "from": "nbest_classes_int32",
        },
        "nbest_embedding": {
            "L2": 5e-06,
            "activation": None,
            "class": "linear",
            "from": "nbest_classes_sparse",
            "n_out": 128,
            "reuse_params": "output/rec/embedding",
            "with_bias": False,
        },
        "nbest_mask_embedding": {
            "axes": "T",
            "class": "pad",
            "from": "nbest_embedding",
            "mode": "constant",
            "padding": (1, 0),
            "value": 0,
        },
        "nbest_label_lm_1": {
            "L2": 5e-06,
            "activation": "tanh",
            "class": "linear",
            "dropout": 0.25,
            "from": "nbest_mask_embedding",
            "n_out": 640,
            "reuse_params": "output/rec/label_lm_1",
        },
        "nbest_label_lm_2": {
            "L2": 5e-06,
            "activation": "tanh",
            "class": "linear",
            "dropout": 0.25,
            "from": "nbest_label_lm_1",
            "n_out": 640,
            "reuse_params": "output/rec/label_lm_2",
        },
        "nbest_MBR_loss": {
            "L2": 5e-06,
            "blank_label": 0,
            "class": "iterative_nbest_mbr",
            "dropout": 0.25,
            "from": from_list + [
                "nbest_label_lm_2",
                "nbest_classes_sparse",
                "nbest_score_float32",
                "nbest_risk_float32",
                "nbest_classes_size_int32",
            ],
            "loss": "as_is",
            "nbest": 4,
            "renorm_scale": 2.5,
            "reuse_joint": ("_joint", 1024, "tanh"),
            "reuse_output": ("_output", 88),
            "reuse_params": {
                "map": {
                    "W_joint": "output/rec/joint_encoding",
                    "W_output": "output/rec/output",
                    "b_joint": "output/rec/joint_encoding",
                    "b_output": "output/rec/output",
                }
            },
            "rnnt_loss_scale": 0.05,
            "use_nbest_score": True,
        },
    })


def make_conformer_transducer_model(
    num_outputs: int,
    conformer_args: Optional[Dict] = None,
    output_args: Optional[Dict] = None,
    conformer_type: str = "wei",
    specaug_old: Optional[Dict[str, Any]] = None,
    specaug_stft: Optional[Dict[str, Any]] = None,
    recognition: bool = False,
) -> Tuple[Dict, Union[str, List[str]]]:
    network = {}
    from_list = ["data"]

    if recognition:
        python_code = []
    else:
        if specaug_stft is not None:
            frame_size = specaug_stft.pop("frame_size", 200)
            frame_shift = specaug_stft.pop("frame_shift", 80)
            fft_size = specaug_stft.pop("fft_size", 256)

            specaug_stft_args = {
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
                **specaug_stft,
            }

            # Add STFT layer
            network["stft"] = {
                "class": "stft",
                "from": ["data"],
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "fft_size": fft_size,
            }
            from_list = ["stft"]

            from_list, python_code = add_specaug_layer_stft(network, from_list=from_list, **specaug_stft_args)

            # Add iSTFT layer
            network["istft"] = {
                "class": "istft",
                "from": from_list,
                "frame_size": frame_size,
                "frame_shift": frame_shift,
                "fft_size": fft_size,
            }
        elif specaug_old is not None:

            sort_layer2 = specaug_old.pop("sort_layer2", False)
            specaug_func = add_specaug_layer_sort_layer2 if sort_layer2 else add_specaug_layer
            specaug_old_args = {
                "max_time_num": 1,
                "max_time": 15,
                "max_feature_num": 5,
                "max_feature": 4,
                **specaug_old,
            }
            from_list, python_code = specaug_func(network, from_list=from_list, **specaug_old_args)      
        elif specaug_old is False:
            python_code = []  # no specaugment
        else:
            from_list, python_code = add_specaug_layer_v2(network, from_list=from_list)

    if conformer_type == "wei":
        network, from_list = add_vgg_stack_wei(network, from_list)
        conformer_args_full = {
            "pos_enc_clip": 32,
            "batch_norm_fix": True,
            "switch_conv_mhsa_module": True,
            "l2": 5e-06,
            "dropout": 0.1,
            **(conformer_args or {}),
        }
        network, from_list = add_conformer_stack_wei(network, from_list, **conformer_args_full)
    else:
        raise NotImplementedError

    network["encoder"] = {
        "class": "copy",
        "from": from_list,
    }
    if output_args.get("transducer_training_stage", "viterbi") == "viterbi":
        add_transducer_viterbi_output_layer(
            network, from_list="encoder", num_outputs=num_outputs, recognition=recognition,
            **{**conformer_args_full, **(output_args or {})}
        )
        if not recognition:
            python_code += [segmental_loss]
            network.update({
                "enc_output": {
                    "class": "softmax",
                    "from": "encoder",
                    "loss": "ce",
                    "loss_opts": {"focal_loss_factor": 1.0},
                },
                "enc_output_loss": {
                    "class": "softmax",
                    "from": "conformer_6_output",
                    "loss": "ce",
                    "loss_opts": {"focal_loss_factor": 1.0},
                    "loss_scale": 0.3,
                },
            })
    elif output_args.get("transducer_training_stage", "viterbi") == "fullsum":
        add_transducer_fullsum_output_layer(
            network, from_list="encoder", num_outputs=num_outputs, recognition=recognition,
            **{**conformer_args_full, **(output_args or {})}
        )
        if recognition:
            python_code.append(subtract_ilm)
        else:
            with open(os.path.join(os.path.dirname(__file__), "returnn_layers_compressed_concat.py")) as f:
                compressed_concat_layer_str = f.read()
            python_code += [
                compressed_concat_layer_str,
                rnnt_loss_compressed,
            ]
    elif output_args.get("transducer_training_stage", "viterbi") == "mbr":
        add_transducer_fullsum_output_layer(  # TODO: needs changes for mbr?
            network, from_list="encoder", num_outputs=num_outputs, recognition=recognition,
            **{**conformer_args_full, **(output_args or {})}
        )
        if recognition:
            python_code.append(subtract_ilm)
        else:
            add_transducer_mbr_layers(network, from_list=["encoder"])
            with open(os.path.join(os.path.dirname(__file__), "returnn_layers_compressed_concat.py")) as f:
                compressed_concat_layer_str = f.read()
            with open(os.path.join(os.path.dirname(__file__), "returnn_layers_nbest_mbr_loss.py")) as f:
                nbest_mbr_loss_layer_str = f.read()
            python_code += [
                compressed_concat_layer_str,
                rnnt_loss_compressed,
                nbest_mbr_loss_layer_str,
            ]
    if not recognition:
        network.update({
            "mask_flag": {
                "class": "compare",
                "from": "data:classes",
                "kind": "not_equal",
                "value": 0,
            },
            "mask_label": {
                "class": "masked_computation",
                "from": "data:classes",
                "mask": "mask_flag",
                "unit": {"class": "copy"},
            },
        })
        if output_args.get("transducer_training_stage", "viterbi") == "viterbi":
            network["encoder"] = {
                "class": "reinterpret_data",
                "from": from_list,
                "size_base": "data:classes",
            }

    return network, python_code
