from typing import Dict, List, Optional, Tuple
from i6_core.returnn import CodeWrapper
from i6_experiments.users.berger.network.helpers.mlp import add_feed_forward_stack
from i6_experiments.users.berger.network.helpers.output import add_softmax_output
from i6_experiments.users.berger.network.helpers.compressed_input import (
    compressed_add_code,
    compressed_concat_code,
    compressed_multiply_code,
)
from enum import Enum, auto


class ILMMode(Enum):
    ZeroEnc = auto()
    ZeroEncInclBlank = auto()


def add_context_label_sequence_blank(
    network: Dict,
    base_labels: str = "data:classes",
    blank_index: int = 0,
) -> Tuple[str, str]:
    # Example:
    # Classes = ..AB..C.D.
    # Then base labels should be ABCD
    # And corresponding mask is 0011001010

    # True at the positions of each non-blank label
    # 0011001010
    network["mask_non_blank"] = {
        "class": "compare",
        "from": base_labels,
        "kind": "not_equal",
        "value": blank_index,
    }

    # Base labels to be expanded to predecessor sequence
    # ABCD
    network["pred_labels"] = {
        "class": "masked_computation",
        "from": base_labels,
        "mask": "mask_non_blank",
        "unit": {"class": "copy"},
    }

    return "pred_labels", "mask_non_blank"


def add_context_label_sequence_noblank(
    network: Dict,
    num_outputs: int,
    nonword_labels: List[int],
    base_labels: str = "data:classes",
):
    # Example:
    # Classes = AAABBCCCDD
    # Let A be a nonword label, e.g. silence
    # Then base labels should be BCD
    # And corresponding mask is 0001010010

    # [
    network["boundary_symbol"] = {
        "class": "eval",
        "from": base_labels,
        "eval": f"tf.ones_like(source(0)) * {num_outputs}",
    }

    # True at positions of non-word classes
    # 1110000000
    eval_str = f"tf.math.not_equal(source(0), {nonword_labels[0]})"
    for label in nonword_labels[1:]:
        eval_str = f"tf.math.logical_and(tf.math.not_equal(source(0), {label}), {eval_str})"
    network["mask_no_nonword"] = {
        "class": "eval",
        "from": base_labels,
        "eval": eval_str,
        "out_type": {"dtype": "bool"},
    }

    # Non-word classes replaced by boundary symbol
    # [[[BBCCCDD
    network["targets_no_nonword"] = {
        "class": "switch",
        "condition": "mask_no_nonword",
        "true_from": base_labels,
        "false_from": "boundary_symbol",
    }
    network["targets_no_nonword_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "targets_no_nonword",
        "increase_sparse_dim": 1,
    }

    # Shift targets forward by one
    #  AAABBCCCD
    network["targets_shifted"] = {
        "class": "shift_axis",
        "from": base_labels,
        "axis": "T",
        "amount": 1,
        "pad": False,
        "adjust_size_info": True,
    }

    # Pad by adding boundary symbol at the start
    # [AAABBCCCD
    network["targets_shifted_padded"] = {
        "class": "pad",
        "from": "targets_shifted",
        "padding": (1, 0),
        "value": num_outputs,
        "axes": "T",
        "mode": "constant",
    }
    network["targets_padded_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "targets_shifted_padded",
        "size_base": base_labels,
        "increase_sparse_dim": 1,
    }

    # True at the first position of each label segment
    # 1001010010
    network["mask_first_label"] = {
        "class": "compare",
        "from": [base_labels, "targets_padded_reinterpret"],
        "kind": "not_equal",
    }

    # Base labels to be expanded to predecessor sequence
    # [BCD
    network["pred_labels"] = {
        "class": "masked_computation",
        "from": "targets_no_nonword_reinterpret",
        "mask": "mask_first_label",
        "unit": {"class": "copy"},
    }

    # Pad by boundary label, the pad is the predecessor of the first label
    # [[BCD
    network["pred_labels_padded"] = {
        "class": "pad",
        "from": "pred_labels",
        "padding": (1, 0),
        "axes": "T",
        "value": num_outputs,
        "mode": "constant",
    }

    return "pred_labels_padded", "mask_first_label"


def add_dec_ffnn_stack(
    output_unit: dict,
    context_labels: str,
    embedding_size: int,
    dec_mlp_args: dict,
) -> str:
    output_unit["context_embedding"] = {
        "class": "linear",
        "from": f"base:{context_labels}",
        "n_out": embedding_size,
        "with_bias": False,
        "L2": dec_mlp_args.get("l2", 5e-06),
    }

    output_unit["context_embedding_padded"] = {
        "class": "pad",
        "from": "context_embedding",
        "padding": (1, 0),
        "axes": "T",
        "value": 0,
        "mode": "constant",
    }

    decoder_ff = add_feed_forward_stack(
        output_unit, from_list="context_embedding_padded", name="dec_ff", **dec_mlp_args
    )

    return decoder_ff


def add_context_1_decoder(
    network: Dict,
    num_outputs: int,
    context_labels: str,
    mask_non_blank: str,
    encoder: str = "encoder",
    embedding_size: int = 128,
    dec_mlp_args: Dict = {},
    joint_mlp_args: Dict = {},
    combination_mode: Optional[str] = "add",
    output_args: Dict = {},
) -> Tuple[List[str], Dict]:
    output_unit = {}

    decoder_ff = add_dec_ffnn_stack(output_unit, context_labels, embedding_size, dec_mlp_args)

    output_unit["mask_non_blank_shifted"] = {
        "class": "shift_axis",
        "from": f"base:{mask_non_blank}",
        "axis": "T",
        "amount": 1,
        "pad": True,
    }

    output_unit["decoder"] = {
        "class": "copy",
        "from": decoder_ff,
    }

    output_unit["decoder_unmasked"] = {
        "class": "unmask",
        "from": "decoder",
        "mask": "mask_non_blank_shifted",
        "skip_initial": True,  # Replace by adding initial output in decoder?
    }

    # Needed?
    output_unit["decoder_unmasked_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "decoder_unmasked",
        "size_base": "data:classes",
    }

    joint_input = ["data:source", "decoder_unmasked_reinterpret"]
    if combination_mode is None or combination_mode == "concat":
        output_unit["joint_input"] = {
            "class": "copy",
            "from": joint_input,
        }
    else:
        output_unit["joint_input"] = {
            "class": "combine",
            "from": joint_input,
            "kind": combination_mode,
        }

    joint_output = add_feed_forward_stack(output_unit, from_list="joint_input", name="joint_ff", **joint_mlp_args)

    add_softmax_output(
        output_unit,
        from_list=joint_output,
        name="output",
        num_outputs=num_outputs,
        target="classes",
        **output_args,
    )

    network["output"] = {
        "class": "rec",
        "from": encoder,
        # only relevant for beam_search: e.g. determine length by targets
        "cheating": False,
        "target": "classes",
        "unit": output_unit,
    }

    return joint_output, output_unit


def add_context_1_decoder_recog(
    network: Dict,
    num_outputs: int,
    blank_idx: int = 0,
    encoder: str = "encoder",
    embedding_size: int = 128,
    dec_mlp_args: Dict = {},
    joint_mlp_args: Dict = {},
    ilm_scale: float = 0.0,
    ilm_mode: ILMMode = ILMMode.ZeroEncInclBlank,
    combination_mode: Optional[str] = "add",
):
    output_unit = {}

    output_unit["output_choice"] = {
        "class": "choice",
        "from": "output_sub_ilm" if ilm_scale else "output",
        "input_type": "log_prob",
        "target": "classes",
        "beam_size": 1,
        "initial_output": num_outputs,
    }
    context_label = "output_choice"

    output_unit["context_embedding"] = {
        "class": "linear",
        "from": f"prev:{context_label}",
        "n_out": embedding_size,
        "with_bias": False,
        "initial_output": None,
        "safe_embedding": True,
    }

    decoder_ff = add_feed_forward_stack(output_unit, from_list="context_embedding", name="dec_ff", **dec_mlp_args)

    output_unit["decoder"] = {
        "class": "copy",
        "from": decoder_ff,
    }

    joint_input = ["data:source", "decoder"]
    if combination_mode is None or combination_mode == "concat":
        output_unit["joint_input"] = {
            "class": "copy",
            "from": joint_input,
        }
    else:
        output_unit["joint_input"] = {
            "class": "combine",
            "from": joint_input,
            "kind": combination_mode,
        }

    joint_output = add_feed_forward_stack(output_unit, from_list="joint_input", name="joint_ff", **joint_mlp_args)

    output_unit["output"] = {
        "class": "linear",
        "from": joint_output,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    if ilm_scale:
        if ilm_mode == ILMMode.ZeroEnc or ilm_mode == ILMMode.ZeroEncInclBlank:
            output_unit["zero_enc"] = {"class": "eval", "from": "data:source", "eval": "source(0) * 0"}
            joint_input_ilm = ["zero_enc", "decoder"]
        else:
            raise NotImplementedError
        if combination_mode is None or combination_mode == "concat":
            output_unit["joint_input_ilm"] = {
                "class": "copy",
                "from": joint_input_ilm,
            }
        else:
            output_unit["joint_input_ilm"] = {
                "class": "combine",
                "from": joint_input_ilm,
                "kind": combination_mode,
            }
        joint_output_ilm = add_feed_forward_stack(
            output_unit, from_list="joint_input_ilm", name="joint_ff_ilm", reuse_from_name="joint_ff", **joint_mlp_args
        )

        output_unit["ilm"] = {
            "class": "linear",
            "from": joint_output_ilm,
            "activation": "log_softmax",
            "n_out": num_outputs,
            "reuse_params": "output",
        }

        if ilm_mode == ILMMode.ZeroEncInclBlank:
            ilm_layer = "ilm"
        else:
            assert blank_idx == 0, "Blank idx != 0 not implemented for ilm"
            # Set p(blank) = 1 and re-normalize the non-blank probs
            # so we want P'[b, 0] = 1, sum(P'[b, 1:]) = 1, given a normalized tensor P, i.e. sum(P[b, :]) = 1
            # in log space logP'[b, 0] = 0, sum(exp(logP'[b, 1:])) = 1
            # so set logP'[b, 1:] <- logP[b, 1:] - log(1 - exp(P[b, 0]))
            # then sum(exp(logP'[b, 1:])) = sum(P[1:] / (1 - exp(P[b, 0]))) = sum(P[b, 1:]) / sum(b, P[1:]) = 1
            output_unit["ilm_renorm"] = {
                "class": "eval",
                "from": ["ilm"],
                "eval": "tf.concat([tf.zeros(tf.shape(source(0)[:, :1])), source(0)[:, 1:] - tf.math.log(1.0 - tf.exp(source(0)[:, :1]))], axis=-1)",
            }
            ilm_layer = "ilm_renorm"

        output_unit["output_sub_ilm"] = {
            "class": "eval",
            "from": ["output", ilm_layer],
            "eval": f"source(0) - {ilm_scale} * source(1)",
        }

    network["output"] = {
        "class": "rec",
        "from": encoder,
        # only relevant for beam_search: e.g. determine length by targets
        "cheating": False,
        "target": "classes",
        "unit": output_unit,
    }

    return joint_output, output_unit


def add_precomputed_context_1_decoder_recog(
    network: Dict,
    num_outputs: int,
    blank_idx: int = 0,
    encoder: str = "encoder",
    embedding_size: int = 128,
    dec_mlp_args: Dict = {},
    joint_mlp_args: Dict = {},
    ilm_scale: float = 0.0,
    ilm_mode: ILMMode = ILMMode.ZeroEncInclBlank,
    combination_mode: Optional[str] = "add",
):
    output_unit = {}

    # [V-1]
    output_unit["all_context"] = {
        "class": "constant",
        "value": list(range(1, num_outputs)),  # first index: out-of-bounds for all-zero-embedding of init history
        "dtype": "int32",
        "with_batch_dim": True,
        "as_batch": True,
        "sparse_dim": CodeWrapper(f"FeatureDim('label', dimension={num_outputs})"),
    }

    # [V-1, F]
    output_unit["context_embedding"] = {
        "class": "linear",
        "from": "all_context",
        "n_out": embedding_size,
        "with_bias": False,
        "initial_output": None,
    }

    # [V, F]
    output_unit["context_embedding_padded"] = {
        "class": "pad",
        "from": "context_embedding",
        "axes": "B",
        "padding": (1, 0),
        "value": 0,
        "mode": "constant",
    }

    # [V, D]
    decoder_ff = add_feed_forward_stack(
        output_unit, from_list="context_embedding_padded", name="dec_ff", **dec_mlp_args
    )
    output_unit["decoder"] = {
        "class": "copy",
        "from": decoder_ff,
    }

    # [V, T, E]
    output_unit["tile_encoder"] = {
        "class": "eval",
        "from": f"base:base:{encoder}",
        "eval": f"tf.tile(source(0), [{num_outputs}, 1, 1])",
    }

    joint_input = ["tile_encoder", "decoder"]
    if combination_mode is None or combination_mode == "concat":
        # [V, T, E+D]
        output_unit["joint_input"] = {
            "class": "copy",
            "from": joint_input,
        }
    else:
        # [V, T, E]
        output_unit["joint_input"] = {
            "class": "combine",
            "from": joint_input,
            "kind": combination_mode,
        }

    # [V, T, J]
    joint_output = add_feed_forward_stack(output_unit, from_list="joint_input", name="joint_ff", **joint_mlp_args)

    # [V, T, V]
    output_unit["output"] = {
        "class": "linear",
        "from": joint_output,
        "activation": "log_softmax",
        "n_out": num_outputs,
    }

    if ilm_scale:
        if ilm_mode == ILMMode.ZeroEnc or ilm_mode == ILMMode.ZeroEncInclBlank:
            # [V, T, E]
            output_unit["zero_enc"] = {"class": "eval", "from": "tile_encoder", "eval": "source(0) * 0"}
            # [V, T, E+D]
            joint_input_ilm = ["zero_enc", "decoder"]
        else:
            raise NotImplementedError
        if combination_mode is None or combination_mode == "concat":
            # [V, T, E+D]
            output_unit["joint_input_ilm"] = {
                "class": "copy",
                "from": joint_input_ilm,
            }
        else:
            # [V, T, E]
            output_unit["joint_input_ilm"] = {
                "class": "combine",
                "from": joint_input_ilm,
                "kind": combination_mode,
            }

        # [V, T, J]
        joint_output_ilm = add_feed_forward_stack(
            output_unit, from_list="joint_input_ilm", name="joint_ff_ilm", reuse_from_name="joint_ff", **joint_mlp_args
        )

        # [V, T, V]
        output_unit["ilm"] = {
            "class": "linear",
            "from": joint_output_ilm,
            "activation": "log_softmax",
            "n_out": num_outputs,
            "reuse_params": "output",
        }

        if ilm_mode == ILMMode.ZeroEncInclBlank:
            ilm_layer = "ilm"
        else:
            assert blank_idx == 0, "Blank idx != 0 not implemented for ilm"
            # Set p(blank) = 1 and re-normalize the non-blank probs
            # so we want P'[b, 0] = 1, sum(P'[b, 1:]) = 1, given a normalized tensor P, i.e. sum(P[b, :]) = 1
            # in log space logP'[b, 0] = 0, sum(exp(logP'[b, 1:])) = 1
            # so set logP'[b, 1:] <- logP[b, 1:] - log(1 - exp(P[b, 0]))
            # then sum(exp(logP'[b, 1:])) = sum(P[1:] / (1 - exp(P[b, 0]))) = sum(P[b, 1:]) / sum(b, P[1:]) = 1
            output_unit["ilm_renorm"] = {
                "class": "eval",
                "from": "ilm",
                "eval": "tf.concat([tf.zeros(tf.shape(source(0)[:, :1])), source(0)[:, 1:] - tf.math.log(1.0 - tf.exp(source(0)[:, :1]))], axis=-1)",
            }
            ilm_layer = "ilm_renorm"

        # [V, T, V]
        output_unit["output_sub_ilm"] = {
            "class": "eval",
            "from": ["output", ilm_layer],
            "eval": f"source(0) - {ilm_scale} * source(1)",
            "is_output_layer": True,
        }

        out_layer = "output/rec/output_sub_ilm"
    else:
        out_layer = "output"

    network["output"] = {
        "class": "subnetwork",
        "from": encoder,
        "subnetwork": {
            "rec": {
                "class": "subnetwork",
                "from": "data",
                "subnetwork": output_unit,
            },
            "output": {
                "class": "copy",
                "from": "rec",
            },
        },
        "is_output_layer": False,
    }

    network["output_precompute"] = {
        "class": "eval",
        "from": out_layer,
        "eval": f"tf.transpose(tf.reshape(tf.transpose(source(0, auto_convert=False, enforce_batch_major=True), [0, 2, 1]), [1, {num_outputs*num_outputs}, -1]), [0, 2, 1])",
        "is_output_layer": True,
        "out_type": {"shape": (None, num_outputs * num_outputs), "dim": num_outputs * num_outputs},
    }

    return output_unit


def add_context_1_decoder_fullsum(
    network: Dict,
    context_labels: str,
    encoder: str = "encoder",
    embedding_size: int = 128,
    dec_mlp_args: Dict = {},
    joint_mlp_args: Dict = {},
    combination_mode: Optional[str] = "add",
    compress_joint_input: bool = True,
) -> Tuple[List[str], Dict, List]:
    output_unit = {}
    extra_python = []

    decoder_ff = add_dec_ffnn_stack(output_unit, f"base:{context_labels}", embedding_size, dec_mlp_args)

    output_unit["decoder"] = {
        "class": "copy",
        "from": decoder_ff,
    }

    joint_input = [f"base:base:{encoder}", "decoder"]
    if compress_joint_input:
        if combination_mode == "concat":
            output_unit["joint_input"] = {
                "class": "compressed_concat",
                "from": joint_input,
            }
            extra_python.append(compressed_concat_code)
        elif combination_mode == "add":
            output_unit["joint_input"] = {
                "class": "compressed_add",
                "from": joint_input,
            }
            extra_python.append(compressed_add_code)
        elif combination_mode == "multiply":
            output_unit["joint_input"] = {
                "class": "compressed_multiply",
                "from": joint_input,
            }
            extra_python.append(compressed_multiply_code)
    else:
        if combination_mode is None or combination_mode == "concat":
            output_unit["joint_input"] = {
                "class": "copy",
                "from": joint_input,
            }
        else:
            output_unit["joint_input"] = {
                "class": "combine",
                "from": joint_input,
                "kind": combination_mode,
            }

    joint_output = add_feed_forward_stack(output_unit, from_list="joint_input", name="joint_ff", **joint_mlp_args)

    # Match name scope from viterbi model to enable initializing from one
    network["output"] = {
        "class": "subnetwork",
        "from": encoder,
        "subnetwork": {
            "output": {
                "class": "copy",
                "from": "rec",
            },
            "rec": {
                "class": "subnetwork",
                "from": "data",
                "subnetwork": output_unit,
            },
        },
    }

    return joint_output, output_unit, extra_python
