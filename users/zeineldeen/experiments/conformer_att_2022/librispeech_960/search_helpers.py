import copy

from i6_core.returnn.config import CodeWrapper


def add_joint_ctc_att_subnet(
    net,
    att_scale,
    ctc_scale,
    check_repeat_version=1,
    beam_size=12,
    remove_eos=False,
    renorm_after_remove_eos=False,
    in_scale=False,
    comb_score_version=1,
    blank_penalty=None,
):
    """
    Add layers for joint CTC and att search.

    :param dict net: network dict
    :param float att_scale: attention score scale
    :param float ctc_scale: ctc score scale
    """
    net["output"] = {
        "class": "rec",
        "from": "ctc",  # [B,T,V+1]
        "unit": {
            "is_prev_out_not_blank_mask": {
                "class": "compare",
                "kind": "not_equal",
                "from": "prev:output",
                "value": 10025,
            },
            "is_curr_out_not_blank_mask": {
                "class": "compare",
                "kind": "not_equal",
                "from": "output",
                "value": 10025,
            },
            # reinterpreted for target_embed
            "output_reinterpret": {
                "class": "reinterpret_data",
                "from": "output",
                "set_sparse": True,
                "set_sparse_dim": 10025,  # V
            },
            "trigg_att": {
                "class": "subnetwork",
                "from": [],
                "n_out": 10025,
                "name_scope": "",
                "subnetwork": {
                    "target_embed0": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": "base:output_reinterpret",
                        "n_out": 640,
                        "L2": 0.0001,
                        "initial_output": 0,
                    },
                    "_target_embed": {
                        "class": "dropout",
                        "from": "target_embed0",
                        "dropout": 0.1,
                        "dropout_noise_shape": {"*": None},
                    },
                    "target_embed": {
                        "class": "switch",
                        "condition": "base:is_curr_out_not_blank_mask",
                        "true_from": "_target_embed",
                        "false_from": "prev:target_embed",
                    },
                    "s_transformed": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": "s",
                        "n_out": 1024,
                        "L2": 0.0001,
                    },
                    "_accum_att_weights": {
                        "class": "eval",
                        "eval": "source(0) + source(1) * source(2) * 0.5",
                        "from": [
                            "prev:accum_att_weights",
                            "att_weights",
                            "base:base:inv_fertility",
                        ],
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "accum_att_weights": {
                        "class": "switch",
                        "condition": "base:is_prev_out_not_blank_mask",
                        "true_from": "_accum_att_weights",
                        "false_from": "prev:accum_att_weights",
                        "out_type": {"dim": 1, "shape": (None, 1)},
                    },
                    "weight_feedback": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": "prev:accum_att_weights",
                        "n_out": 1024,
                    },
                    "energy_in": {
                        "class": "combine",
                        "kind": "add",
                        "from": [
                            "base:base:enc_ctx",
                            "weight_feedback",
                            "s_transformed",
                        ],
                        "n_out": 1024,
                    },
                    "energy_tanh": {
                        "class": "activation",
                        "activation": "tanh",
                        "from": "energy_in",
                    },
                    "energy": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": "energy_tanh",
                        "n_out": 1,
                        "L2": 0.0001,
                    },
                    "att_weights": {"class": "softmax_over_spatial", "from": "energy"},
                    "att0": {
                        "class": "generic_attention",
                        "weights": "att_weights",
                        "base": "base:base:enc_value",
                    },
                    "att": {
                        "class": "merge_dims",
                        "from": "att0",
                        "axes": "except_batch",
                    },
                    "_s": {
                        "class": "rnn_cell",
                        "unit": "zoneoutlstm",
                        "n_out": 1024,
                        "from": ["prev:target_embed", "prev:att"],
                        "L2": 0.0001,
                        "unit_opts": {
                            "zoneout_factor_cell": 0.15,
                            "zoneout_factor_output": 0.05,
                        },
                        "name_scope": "s/rec",  # compatibility with old models
                        "state": CodeWrapper("tf_v1.nn.rnn_cell.LSTMStateTuple('prev:s_c', 'prev:s_h')"),
                    },
                    "s": {
                        "class": "switch",
                        "condition": "base:is_prev_out_not_blank_mask",
                        "true_from": "_s",
                        "false_from": "prev:s",
                    },
                    "_s_c": {
                        "class": "get_last_hidden_state",
                        "from": "_s",
                        "key": "c",
                        "n_out": 1024,
                    },
                    "s_c": {
                        "class": "switch",
                        "condition": "base:is_prev_out_not_blank_mask",
                        "true_from": "_s_c",
                        "false_from": "prev:s_c",
                    },
                    "_s_h": {
                        "class": "get_last_hidden_state",
                        "from": "_s",
                        "key": "h",
                        "n_out": 1024,
                    },
                    "s_h": {
                        "class": "switch",
                        "condition": "base:is_prev_out_not_blank_mask",
                        "true_from": "_s_h",
                        "false_from": "prev:s_h",
                    },
                    "readout_in": {
                        "class": "linear",
                        "activation": None,
                        "with_bias": True,
                        "from": ["s", "prev:target_embed", "att"],
                        "n_out": 1024,
                        "L2": 0.0001,
                    },
                    "readout": {
                        "class": "reduce_out",
                        "from": "readout_in",
                        "num_pieces": 2,
                        "mode": "max",
                    },
                    "output_prob": {
                        "class": "softmax",
                        "from": "readout",
                        "target": "bpe_labels",
                    },
                    "output": {"class": "copy", "from": "output_prob"},
                },
            },
            "att_log_scores_": {
                "class": "activation",
                "activation": "safe_log",
                "from": "trigg_att",
            },
            "att_log_scores": {
                "class": "switch",
                "condition": "is_prev_out_not_blank_mask",
                "true_from": "att_log_scores_",
                "false_from": "prev:att_log_scores",
            },
            "ctc_log_scores": {
                "class": "activation",
                "activation": "safe_log",
                "from": "data:source",
            },  # [B,V+1]
            # log p_comb_sigma = log p_att_sigma + log p_ctc_sigma (using only labels without blank)
            "ctc_log_scores_slice": {
                "class": "slice",
                "from": "ctc_log_scores",
                "axis": "f",
                "slice_start": 0,
                "slice_end": 10025,  # excluding blank
            },  # [B,V]
            "ctc_log_scores_norm": {
                "class": "reduce",
                "mode": "logsumexp",
                "from": "ctc_log_scores_slice",
                "axis": "f",
            },
            # renormalize label probs without blank
            "ctc_log_scores_renorm": {
                "class": "combine",
                "kind": "sub",
                "from": ["ctc_log_scores_slice", "ctc_log_scores_norm"],
            },
            "scaled_att_log_scores": {
                "class": "eval",
                "from": "att_log_scores",
                "eval": f"{att_scale} * source(0)",
            },
            "scaled_ctc_log_scores": {
                "class": "eval",
                "from": "ctc_log_scores_renorm",
                "eval": f"{ctc_scale} * source(0)",
            },
            "combined_att_ctc_scores": {
                "class": "combine",
                "kind": "add",
                "from": ["scaled_att_log_scores", "scaled_ctc_log_scores"],
            },  # [B,V]
            # log p_ctc_sigma' (blank | ...)
            "blank_log_prob": {
                "class": "gather",
                "from": "ctc_log_scores",
                "position": 10025,
                "axis": "f",
            },  # [B]
            # p_ctc_sigma' (blank | ...)
            "blank_prob": {"class": "gather", "from": "data:source", "position": 10025, "axis": "f"},
            # p_comb_sigma' for labels which is defined as:
            # (1 - p_ctc_sigma'(blank | ...)) * p_comb_sigma(label | ...)
            # here is not log-space
            "one": {"class": "constant", "value": 1.0},
            "1_minus_blank": {
                "class": "combine",
                "kind": "sub",
                "from": ["one", "blank_prob"],
            },
            "1_minus_blank_log": {
                "class": "activation",
                "activation": "safe_log",
                "from": "1_minus_blank",
            },
            "scaled_1_minus_blank_log": {
                "class": "eval",
                "from": "1_minus_blank_log",
                "eval": f"{ctc_scale} * source(0)",
            },
            "p_comb_sigma_prime_label": {
                "class": "combine",
                "kind": "add",
                "from": ["scaled_1_minus_blank_log", "combined_att_ctc_scores"],
            },
            "scaled_blank_log_prob": {
                "class": "eval",
                "from": "blank_log_prob",
                "eval": f"{ctc_scale} * source(0)",
            },
            "scaled_blank_log_prob_expand": {
                "class": "expand_dims",
                "from": "scaled_blank_log_prob",
                "axis": "f",
            },  # [B,1]
            "p_comb_sigma_prime": {
                "class": "concat",
                "from": [("p_comb_sigma_prime_label", "f"), ("scaled_blank_log_prob_expand", "f")],
            },  # [B,V+1]
            "output": {
                "class": "choice",
                "target": "bpe_labels_w_blank" if ctc_scale > 0.0 else "bpe_labels",
                "beam_size": beam_size,
                "from": "p_comb_sigma_prime" if ctc_scale > 0.0 else "combined_att_ctc_scores",
                "input_type": "log_prob",
                "initial_output": 0,
                "length_normalization": False,
            },
        },
        "target": "bpe_labels_w_blank" if ctc_scale > 0.0 else "bpe_labels",
    }
    if check_repeat_version:
        net["output"]["unit"]["not_repeat_mask"] = {
            "class": "compare",
            "from": ["output", "prev:output"],
            "kind": "not_equal",
        }
        net["output"]["unit"]["is_curr_out_not_blank_mask_"] = copy.deepcopy(
            net["output"]["unit"]["is_curr_out_not_blank_mask"]
        )
        net["output"]["unit"]["is_curr_out_not_blank_mask"] = {
            "class": "combine",
            "kind": "logical_and",
            "from": ["is_curr_out_not_blank_mask_", "not_repeat_mask"],
        }
        net["output"]["unit"]["is_curr_out_not_blank_mask"]["initial_output"] = True
        net["output"]["unit"]["is_prev_out_not_blank_mask"] = {
            "class": "copy",
            "from": "prev:is_curr_out_not_blank_mask",
        }
    if remove_eos:
        net["output"]["unit"]["att_scores_wo_eos"] = {
            "class": "eval",
            "eval": CodeWrapper("update_tensor_entry"),
            "from": "trigg_att",
        }
        if renorm_after_remove_eos:
            net["output"]["unit"]["att_log_scores_"] = copy.deepcopy(net["output"]["unit"]["att_log_scores"])
            net["output"]["unit"]["att_log_scores_"]["from"] = "att_scores_wo_eos"
            net["output"]["unit"]["att_log_scores_norm"] = {
                "class": "reduce",
                "mode": "logsumexp",
                "from": "att_log_scores_",
                "axis": "f",
            }
            net["output"]["unit"]["att_log_scores"] = {
                "class": "combine",
                "kind": "sub",
                "from": ["att_log_scores_", "att_log_scores_norm"],
            }
        else:
            net["output"]["unit"]["att_log_scores_"]["from"] = "att_scores_wo_eos"
    if in_scale:
        net["output"]["unit"]["scaled_blank_prob"] = {
            "class": "eval",
            "from": "blank_prob",
            "eval": f"source(0) ** {ctc_scale}",
        }
        # 1 - p_ctc(...)^scale
        net["output"]["unit"]["1_minus_blank"] = {
            "class": "combine",
            "kind": "sub",
            "from": ["one", "scaled_blank_prob"],
        }
        # no scaling outside
        net["output"]["unit"]["scaled_1_minus_blank_log"] = {"class": "copy", "from": "1_minus_blank_log"}
    if comb_score_version == 2:
        net["output"]["unit"]["p_comb_sigma_prime"]["from"][0] = ("combined_att_ctc_scores", "f")
    if comb_score_version == 3:
        # normalize p_comb_sigma
        net["output"]["unit"]["combined_att_ctc_scores_norm"] = {
            "class": "reduce",
            "mode": "logsumexp",
            "from": "combined_att_ctc_scores",
            "axis": "f",
        }
        net["output"]["unit"]["combined_att_ctc_scores_renorm"] = {
            "class": "combine",
            "kind": "sub",
            "from": ["combined_att_ctc_scores", "combined_att_ctc_scores_norm"],
        }
        net["output"]["unit"]["p_comb_sigma_prime_label"] = {
            "class": "combine",
            "kind": "add",
            "from": ["scaled_1_minus_blank_log", "combined_att_ctc_scores_renorm"],
        }
    if blank_penalty:
        net["output"]["unit"]["scaled_blank_log_prob_expand_"] = copy.deepcopy(
            net["output"]["unit"]["scaled_blank_log_prob_expand"]
        )
        net["output"]["unit"]["scaled_blank_log_prob_expand"] = {
            "class": "eval",
            "from": "scaled_blank_log_prob_expand_",
            "eval": f"source(0) - {blank_penalty}",
        }


def add_filter_blank_and_merge_labels_layers(net):
    """
    Add layers to filter out blank and merge repeated labels of a CTC output sequence.
    :param dict net: network dict
    """

    net["out_best_"] = {"class": "decide", "from": "output", "target": "bpe_labels_w_blank"}
    net["out_best"] = {
        "class": "reinterpret_data",
        "from": "out_best_",
        "set_sparse_dim": 10025,
    }
    # shift to the right to create a boolean mask later where it is true if the previous label is equal
    net["shift_right"] = {
        "class": "shift_axis",
        "from": "out_best",
        "axis": "T",
        "amount": 1,
        "pad_value": -1,  # to have always True at the first pos
    }
    # reinterpret time axis to work with following layers
    net["out_best_time_reinterpret"] = {
        "class": "reinterpret_data",
        "from": "out_best",
        "size_base": "shift_right",  # [B,T|shift_axis]
    }
    net["unique_mask"] = {
        "class": "compare",
        "kind": "not_equal",
        "from": ["out_best_time_reinterpret", "shift_right"],
    }
    net["non_blank_mask"] = {
        "class": "compare",
        "from": "out_best_time_reinterpret",
        "value": 10025,
        "kind": "not_equal",
    }
    net["out_best_mask"] = {
        "class": "combine",
        "kind": "logical_and",
        "from": ["unique_mask", "non_blank_mask"],
    }
    net["out_best_wo_blank"] = {
        "class": "masked_computation",
        "from": "out_best_time_reinterpret",
        "mask": "out_best_mask",
        "unit": {"class": "copy"},
        "target": "bpe_labels",
    }
    net["edit_distance"] = {
        "class": "copy",
        "from": "out_best_wo_blank",
        "only_on_search": True,
        "loss": "edit_distance",
        "target": "bpe_labels",
    }


def create_ctc_greedy_decoder(net):
    """
    Create a greedy decoder for CTC.

    :param dict net: network dict
    """

    # time-sync search
    assert net["output"]["class"] == "rec"
    net["output"]["from"] = "ctc"  # [B,T,V+1]
    net["output"]["target"] = "bpe_labels_w_blank"

    # used for label-sync search
    net["output"]["unit"].pop("end", None)
    net["output"].pop("max_seq_len", None)

    # can be also done simply via tf.argmax but it is anw fast
    net["output"]["unit"]["output"] = {
        "class": "choice",
        "target": "bpe_labels_w_blank",
        "beam_size": 1,
        "from": "data:source",
        "initial_output": 0,
    }


def update_tensor_entry(source, **kwargs):
    import tensorflow as tf

    tensor = source(0)
    batch_size = tf.shape(tensor)[0]
    indices = tf.range(batch_size)  # [0, 1, ..., batch_size - 1]
    indices = tf.expand_dims(indices, axis=1)  # [[0], [1], ..., [batch_size - 1]]
    zeros = tf.zeros_like(indices)
    indices = tf.concat([indices, zeros], axis=1)  # [[0, 0], [1, 0], ..., [batch_size - 1, 0]]
    updates = tf.zeros([batch_size])
    return tf.tensor_scatter_nd_update(tensor, indices, updates)
