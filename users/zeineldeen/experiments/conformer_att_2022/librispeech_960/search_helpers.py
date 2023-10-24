import copy

from i6_core.returnn.config import CodeWrapper


def add_joint_ctc_att_subnet(
    net,
    att_scale,
    ctc_scale,
    ctc_prior_scale,
    beam_size=12,
    remove_eos=False,
    renorm_after_remove_eos=False,
    only_scale_comb=True,
    comb_score_version=1,
    scale_outside=False,
):
    """
    Add layers for joint CTC and att search.

    :param dict net: network dict
    :param float att_scale: attention score scale
    :param float ctc_scale: ctc score scale
    """

    ctc_probs_scale = 1.0 if only_scale_comb else ctc_scale
    one_minus_term_scale = 1.0 if scale_outside else ctc_scale

    net["output"] = {
        "class": "rec",
        "from": "ctc",  # [B,T,V+1]
        "unit": {
            "not_repeat_mask": {
                "class": "compare",
                "from": ["output", "prev:output"],
                "kind": "not_equal",
            },
            "is_curr_out_not_blank_mask": {
                "class": "compare",
                "kind": "not_equal",
                "from": "output",
                "value": 10025,
            },
            "is_prev_out_not_blank_mask": {
                "class": "compare",
                "kind": "not_equal",
                "from": "prev:output",
                "value": 10025,
            },  # TODO: try also with prev:is_curr_out_not_blank_mask and intial_output True
            "curr_mask": {
                "class": "combine",
                "kind": "logical_and",
                "from": ["is_curr_out_not_blank_mask", "not_repeat_mask"],
                "initial_output": True,
            },
            "prev_mask": {
                "class": "copy",
                "from": "prev:curr_mask",
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
                        "condition": "base:curr_mask",
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
                        "condition": "base:prev_mask",
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
                        "condition": "base:prev_mask",
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
                        "condition": "base:prev_mask",
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
                        "condition": "base:prev_mask",
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
            "att_log_scores": {
                "class": "activation",
                "activation": "safe_log",
                "from": "trigg_att",
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
            # ----------------------------- #
            "vocab_range": {"class": "range", "limit": 10025},
            "prev_output_reinterpret": {
                "class": "reinterpret_data",
                "from": "prev:output",
                "set_sparse": True,
                "set_sparse_dim": 10025,
            },
            "prev_repeat_mask": {
                "class": "compare",
                "from": ["prev_output_reinterpret", "vocab_range"],
                "kind": "equal",  # always False for blank
            },
            # ----------------------------- #
            # p_ctc_sigma' (blank | ...)
            "blank_prob": {"class": "gather", "from": "data:source", "position": 10025, "axis": "f"},
            "scaled_blank_prob": {
                "class": "eval",
                "from": "blank_prob",
                "eval": f"source(0) ** {one_minus_term_scale}",
            },
            "one": {"class": "constant", "value": 1.0},
            "prev_ctc_log_scores": {
                "class": "gather",
                "from": "ctc_log_scores",
                "position": "prev:output",
                "axis": "f",
            },
            "scaled_prev_ctc_log_scores": {
                "class": "eval",
                "from": "prev_ctc_log_scores",
                "eval": f"{one_minus_term_scale} * source(0)",
            },
            "scaled_prev_ctc_scores": {
                "class": "activation",
                "activation": "safe_exp",
                "from": "scaled_prev_ctc_log_scores",
            },
            "repeat_prob_term": {
                "class": "switch",
                "condition": "is_prev_out_not_blank_mask",
                "true_from": "scaled_prev_ctc_scores",  # p(label:=prev:label|...)
                "false_from": 0.0,
            },
            "1_minus_term_": {
                "class": "combine",
                "kind": "sub",
                "from": ["one", "scaled_blank_prob"],
            },
            "1_minus_term": {
                "class": "combine",
                "kind": "sub",
                "from": ["1_minus_term_", "repeat_prob_term"],
            },
            "1_minus_term_log": {
                "class": "activation",
                "activation": "safe_log",
                "from": "1_minus_term",
            },
            # [1 - P_ctc(blank|...) - P_ctc(label:=prev:label|...)] * P_att(label|...)  # prev:label != blank
            "p_comb_sigma_prime_label": {
                "class": "combine",
                "kind": "add",
                "from": ["1_minus_term_log", "combined_att_ctc_scores"],
            },
            # ----------------------------- #
            "scaled_blank_log_prob": {
                "class": "eval",
                "from": "blank_log_prob",
                "eval": f"{ctc_probs_scale} * source(0)",
            },
            "scaled_blank_log_prob_expand": {
                "class": "expand_dims",
                "from": "scaled_blank_log_prob",
                "axis": "f",
            },  # [B,1]
            "scaled_ctc_log_scores_slice": {
                "class": "eval",
                "from": "ctc_log_scores_slice",
                "eval": f"{ctc_probs_scale} * source(0)",
            },
            "scaled_label_score": {
                "class": "switch",
                "condition": "prev_repeat_mask",
                "true_from": "scaled_ctc_log_scores_slice",  # log P_ctc(label|...) in case label (not blank) is repeated
                "false_from": "p_comb_sigma_prime_label",  # [1 - ...] * p_comb_sigma
            },
            "p_comb_sigma_prime": {
                "class": "concat",
                "from": [("scaled_label_score", "f"), ("scaled_blank_log_prob_expand", "f")],
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

    if comb_score_version == 1:
        pass  # default one
    elif comb_score_version == 2:
        # use only p_comb_sigma
        net["output"]["unit"]["scaled_label_score"]["false_from"] = "combined_att_ctc_scores"
    elif comb_score_version == 3:
        # TODO: this is not properly normalized. It requires renormalizing the comb probs without the repeated label.
        # normalize p_comb_sigma
        net["output"]["unit"]["scaled_ctc_log_scores_exp"] = {
            "class": "activation",
            "activation": "safe_exp",
            "from": "scaled_ctc_log_scores",
        }
        net["output"]["unit"]["combined_att_ctc_scores_norm_"] = {
            "class": "combine",
            "kind": "mul",
            "from": ["trigg_att", "scaled_ctc_log_scores_exp"],
        }
        net["output"]["unit"]["combined_att_ctc_scores_norm"] = {
            "class": "reduce",
            "mode": "sum",
            "from": "combined_att_ctc_scores_norm_",
            "axis": "f",
        }
        net["output"]["unit"]["combined_att_ctc_scores_norm_log"] = {
            "class": "activation",
            "activation": "safe_log",
            "from": "combined_att_ctc_scores_norm",
        }
        net["output"]["unit"]["combined_att_ctc_scores_"] = copy.deepcopy(
            net["output"]["unit"]["combined_att_ctc_scores"]
        )
        net["output"]["unit"]["combined_att_ctc_scores"] = {
            "class": "combine",
            "kind": "sub",
            "from": ["combined_att_ctc_scores_", "combined_att_ctc_scores_norm_log"],
        }
    else:
        raise ValueError(f"invalid comb_score_version {comb_score_version}")

    if scale_outside:
        net["output"]["unit"]["1_minus_term_log_"] = copy.deepcopy(net["output"]["unit"]["1_minus_term_log"])
        net["output"]["unit"]["1_minus_term_log"] = {
            "class": "eval",
            "from": "1_minus_term_log_",
            "eval": f"{ctc_scale} * source(0)",
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


def add_filter_blank_and_merge_labels_layers(net, blank_idx):
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
        "value": blank_idx,
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


def create_ctc_decoder(net, beam_size, ctc_prior_scale, remove_eos=False):
    """
    Create a greedy decoder for CTC.

    :param dict net: network dict
    """

    # time-sync search
    assert net["output"]["class"] == "rec"

    if ctc_prior_scale:
        assert "ctc_log_prior" in net, "ctc log prior layer is not added to network?"
        net["ctc_log_prior_scaled"] = {
            "class": "eval",
            "from": "ctc_log_prior",
            "eval": f"{ctc_prior_scale} * source(0)",
        }
        net["ctc_prior_scaled"] = {"class": "activation", "activation": "safe_exp", "from": "ctc_log_prior_scaled"}
        net["ctc_w_prior"] = {"class": "combine", "kind": "truediv", "from": ["ctc", "ctc_prior_scaled"]}

    net["output"]["from"] = "ctc_w_prior" if ctc_prior_scale else "ctc"  # [B,T,V+1]
    net["output"]["target"] = "bpe_labels_w_blank"

    if remove_eos:
        net["vocab_arange"] = {"class": "range_in_axis", "axis": "F", "from": "ctc"}  # 0...V
        net["ctc_eos_mask"] = {"class": "compare", "from": "vocab_arange", "value": 0, "kind": "not_equal"}
        net["ctc_wo_eos"] = {
            "class": "switch",
            "condition": "ctc_eos_mask",
            "true_from": "ctc_w_prior" if ctc_prior_scale else "ctc",
            "false_from": 1e-20,
        }
        net["output"]["from"] = "ctc_wo_eos"

    # used for label-sync search
    net["output"]["unit"].pop("end", None)
    net["output"].pop("max_seq_len", None)

    # can be also done simply via tf.argmax but it is anw fast
    net["output"]["unit"]["output"] = {
        "class": "choice",
        "target": "bpe_labels_w_blank",
        "beam_size": beam_size,
        "from": "data:source",
        "initial_output": 0,
        "length_normalization": False,
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


def add_ctc_forced_align_for_rescore(net, ctc_prior_scale):
    beam = net["output"]["unit"]["output"]["beam_size"]
    net["extra.search:output"] = copy.deepcopy(net["output"])  # use search in forward
    net["decision"]["from"] = "extra.search:output"
    del net["output"]  # set to ctc scores later and this will be dumped
    net["extra.search:output"]["register_as_extern_data"] = "att_nbest"

    if ctc_prior_scale:
        assert "ctc_log_prior" in net, "ctc log prior layer is not added to network?"
        net["ctc_log_prior_scaled"] = {
            "class": "eval",
            "from": "ctc_log_prior",
            "eval": f"{ctc_prior_scale} * source(0)",
        }
        net["ctc_prior_scaled"] = {"class": "activation", "activation": "safe_exp", "from": "ctc_log_prior_scaled"}
        net["ctc_w_prior"] = {"class": "combine", "kind": "truediv", "from": ["ctc", "ctc_prior_scaled"]}

    net["ctc_align"] = {
        "class": "forced_align",
        "from": "ctc_w_prior" if ctc_prior_scale else "ctc",
        "input_type": "prob",
        "topology": "ctc",
        "align_target": "data:att_nbest",  # [B*Beam,T]
        "target": "bpe_labels_w_blank",
    }
    net["ctc_scores_"] = {"class": "copy", "from": "ctc_align/scores"}  # [B*Beam]
    net["ctc_scores"] = {"class": "split_dims", "from": "ctc_scores_", "axis": "B", "dims": (-1, beam)}  # [B,Beam]
    net["output"] = {"class": "expand_dims", "from": "ctc_scores", "axis": "t"}  # [B,Beam,1|T]


from sisyphus import tk
from typing import Optional, Union, Set
from .default_tools import SCTK_BINARY_PATH


def rescore_att_ctc_search(
    prefix_name,
    returnn_config,
    checkpoint,
    recognition_dataset,
    recognition_reference,
    recognition_bliss_corpus,
    returnn_exe,
    returnn_root,
    mem_rqmt,
    time_rqmt,
    att_scale,
    ctc_scale,
    ctc_prior_scale,
    use_sclite=True,
    rescore_with_ctc: bool = True,
    remove_label: Optional[Union[str, Set[str]]] = None,
):
    """
    Run search for a specific test dataset

    :param str prefix_name:
    :param ReturnnConfig returnn_config:
    :param Checkpoint checkpoint:
    :param returnn_standalone.data.datasets.dataset.GenericDataset recognition_dataset:
    :param Path recognition_reference: Path to a py-dict format reference file
    :param Path returnn_exe:
    :param Path returnn_root:
    :param remove_label: for SearchRemoveLabelJob
    """
    from i6_core.returnn.search import (
        ReturnnSearchJobV2,
        SearchBPEtoWordsJob,
        ReturnnComputeWERJob,
    )
    from i6_core.returnn.forward import ReturnnForwardJob
    from i6_experiments.users.zeineldeen.experiments.conformer_att_2022.librispeech_960.helper_jobs.search import (
        SearchTakeBestRescore,
    )

    assert rescore_with_ctc is True, "Only CTC rescore is supported at the moment"

    att_config = copy.deepcopy(returnn_config)
    att_config.config["search_output_layer"] = "output"  # n-best list
    beam = att_config.config["network"]["output"]["unit"]["output"]["beam_size"]
    att_config.config["batch_size"] = int(att_config.config["batch_size"] * (0.5 if beam > 64 else 1.0))
    search_job = ReturnnSearchJobV2(
        search_data=recognition_dataset.as_returnn_opts(),
        model_checkpoint=checkpoint,
        returnn_config=att_config,
        log_verbosity=5,
        mem_rqmt=mem_rqmt,
        time_rqmt=time_rqmt,
        returnn_python_exe=returnn_exe,
        returnn_root=returnn_root,
    )
    search_job.add_alias(prefix_name + "/search_job")

    ctc_search_config = copy.deepcopy(returnn_config)
    add_ctc_forced_align_for_rescore(ctc_search_config.config["network"], ctc_prior_scale)
    ctc_search_config.config["need_data"] = True
    ctc_search_config.config["target"] = "bpe_labels"
    beam = ctc_search_config.config["network"]["extra.search:output"]["unit"]["output"]["beam_size"]
    ctc_search_config.config["forward_batch_size"] = int(
        ctc_search_config.config["batch_size"] * (0.3 if beam > 32 else 1)
    )
    ctc_search_config.config["eval"] = recognition_dataset.as_returnn_opts()
    forward_job = ReturnnForwardJob(
        model_checkpoint=checkpoint,
        returnn_config=ctc_search_config,
        returnn_root=returnn_root,
        returnn_python_exe=returnn_exe,
        eval_mode=False,
    )
    forward_job.add_alias(prefix_name + "/forward_job")

    search_bpe = search_job.out_search_file
    tk.register_output(prefix_name + "/search_job/search_bpe", search_bpe)
    hdf_scores = forward_job.out_hdf_files["output.hdf"]
    tk.register_output(prefix_name + "/search_job/hdf_scores", hdf_scores)

    search_bpe = SearchTakeBestRescore(
        search_bpe, hdf_scores, scale1=att_scale, scale2=ctc_scale
    ).out_best_search_results
    tk.register_output(prefix_name + "/search_job/comb_search_bpe", search_bpe)

    if remove_label:
        from i6_core.returnn.search import SearchRemoveLabelJob

        search_bpe = SearchRemoveLabelJob(search_bpe, remove_label=remove_label, output_gzip=True).out_search_results

    search_words = SearchBPEtoWordsJob(search_bpe).out_word_search_results
    tk.register_output(prefix_name + "/search_job/comb_search_words", search_words)

    if use_sclite:
        from i6_core.returnn.search import SearchWordsToCTMJob
        from i6_core.corpus.convert import CorpusToStmJob
        from i6_core.recognition.scoring import ScliteJob

        search_ctm = SearchWordsToCTMJob(
            recog_words_file=search_words,
            bliss_corpus=recognition_bliss_corpus,
        ).out_ctm_file

        stm_file = CorpusToStmJob(bliss_corpus=recognition_bliss_corpus).out_stm_path

        sclite_job = ScliteJob(ref=stm_file, hyp=search_ctm, sctk_binary_path=SCTK_BINARY_PATH)
        tk.register_output(prefix_name + "/sclite/wer", sclite_job.out_wer)
        tk.register_output(prefix_name + "/sclite/report", sclite_job.out_report_dir)

    wer = ReturnnComputeWERJob(search_words, recognition_reference)

    tk.register_output(prefix_name + "/wer", wer.out_wer)
    return wer.out_wer
