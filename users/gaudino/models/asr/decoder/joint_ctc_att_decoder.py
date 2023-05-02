"""One pass joint decoder for CTC and attention based models."""

from i6_experiments.users.zeineldeen.modules.network import ReturnnNetwork
from i6_experiments.users.zeineldeen.modules.attention import AttentionMechanism


class CTCAttJointDecoder:
    def __init__(
        self,
        base_model,
        att_scale,
        ctc_scale,
        ctc_greedy_decode,
        joint_ctc_att_decode,
    ):
        self.base_model = base_model  # encoder model
        self.att_scale = att_scale
        self.ctc_scale = ctc_scale

        self.ctc_greedy_decode = ctc_greedy_decode
        self.joint_ctc_att_decode = joint_ctc_att_decode

        # internal variables
        self.network = ReturnnNetwork()
        self.subnet_unit = ReturnnNetwork()
        self.dec_output = None
        self.output_prob = None
        self.decision_layer_name = None

    def add_joint_decoder_subnetwork(self, subnet_unit: ReturnnNetwork):
        subnet_unit._net = (
            {
                "is_not_blank_mask": {
                    "class": "compare",
                    "kind": "not_equal",
                    "from": "prev:output",  # TODO: is this correct?
                    "value": 10025,
                },
                "out_non_blank": {
                    "class": "reinterpret_data",
                    "from": "output",
                    "set_sparse": True,
                    "set_sparse_dim": 10025,  # V
                },
                "trigg_att": {
                    "class": "subnetwork",
                    "from": "out_non_blank",  # reinterpreted for target_embed
                    "n_out": 10025,
                    "name_scope": "",
                    "subnetwork": {
                        "target_embed0": {
                            "class": "linear",
                            "activation": None,
                            "with_bias": False,
                            "from": "data",
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
                            "condition": "base:is_not_blank_mask",
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
                            "condition": "base:is_not_blank_mask",
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
                        "att_weights": {
                            "class": "softmax_over_spatial",
                            "from": "energy",
                        },
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
                            "state": CodeWrapper(
                                "tf_v1.nn.rnn_cell.LSTMStateTuple('prev:s_c', 'prev:s_h')"
                            ),
                        },
                        "s": {
                            "class": "switch",
                            "condition": "base:is_not_blank_mask",
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
                            "condition": "base:is_not_blank_mask",
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
                            "condition": "base:is_not_blank_mask",
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
                },  # [B,V]
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
                "blank_prob": {
                    "class": "gather",
                    "from": "data:source",
                    "position": 10025,
                    "axis": "f",
                },
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
                "p_comb_sigma_prime_label": {
                    "class": "combine",
                    "kind": "add",
                    "from": ["1_minus_blank_log", "combined_att_ctc_scores"],
                },
                "blank_log_prob_expand": {
                    "class": "expand_dims",
                    "from": "blank_log_prob",
                    "axis": "f",
                },
                # [B,1]
                "p_comb_sigma_prime": {
                    "class": "concat",
                    "from": [
                        ("p_comb_sigma_prime_label", "f"),
                        ("blank_log_prob_expand", "f"),
                    ],
                },  # [B,V+1]
                "output": {
                    "class": "choice",
                    "target": "bpe_labels_w_blank",
                    "beam_size": 12,
                    "from": "p_comb_sigma_prime",
                    "input_type": "log_prob",
                    "initial_output": 0,
                },
            },
        )

        # recurrent subnetwork
        dec_output = self.network.add_subnet_rec_layer(
            "output",
            unit=subnet_unit.get_net(),
            target="bpe_labels_w_blank",
            source="ctc",
        )

        return dec_output

    def add_greed_decoder_subnetwork(self, subnet_unit: ReturnnNetwork):
        subnet_unit._net = {
            "output": {
                "class": "choice",
                "target": "bpe_labels_w_blank",
                "beam_size": 1,  # TODO: make it generic. we need recombination?
                "from": "data:source",
                "initial_output": 0,
            }
        }

        # recurrent subnetwork
        dec_output = self.network.add_subnet_rec_layer(
            "output",
            unit=subnet_unit.get_net(),
            target="bpe_labels_w_blank",
            source="ctc",
        )

        return dec_output

    def create_network(self):
        assert not(self.joint_ctc_att_decode and self.ctc_greedy_decode), "Only one of joint_ctc_att_decode and ctc_greedy_decode can be True"
        if self.ctc_greedy_decode:
            self.dec_output = self.add_greed_decoder_subnetwork(self.subnet_unit)
        else:
            self.dec_output = self.add_joint_decoder_subnetwork(self.subnet_unit)

        # Add to Base/Encoder network
        if hasattr(self.base_model, "enc_proj_dim") and self.base_model.enc_proj_dim:
            self.base_model.network.add_copy_layer("enc_ctx", "encoder_proj")
            self.base_model.network.add_split_dim_layer(
                "enc_value",
                "encoder_proj",
                dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads),
            )
        else:
            self.base_model.network.add_linear_layer(
                "enc_ctx",
                "encoder",
                with_bias=True,
                n_out=self.enc_key_dim,
                l2=self.base_model.l2,
            )
            self.base_model.network.add_split_dim_layer(
                "enc_value",
                "encoder",
                dims=(self.att_num_heads, self.enc_value_dim // self.att_num_heads),
            )

        self.base_model.network.add_linear_layer(
            "inv_fertility",
            "encoder",
            activation="sigmoid",
            n_out=self.att_num_heads,
            with_bias=False,
        )

        # time-sync search
        assert self.base_model.network["output"]["class"] == "rec"

        # Decision Postprocessing
        # Remove repeated labels and blanks
        # decision_layer_name = self.base_model.network.add_decide_layer("decision", self.dec_output, target=self.target)

        # filter out blanks from best hyp
        # TODO: we might want to also dump blank for analysis, however, this needs some fix to work.
        self.base_model.network["out_best_"] = {
            "class": "decide",
            "from": "output",
            "target": "bpe_labels_w_blank",
        }
        self.base_model.network["out_best"] = {
            "class": "reinterpret_data",
            "from": "out_best_",
            "set_sparse_dim": 10025,
        }
        # shift to the right to create a boolean mask later where it is true if the previous label is equal
        self.base_model.network["shift_right"] = {
            "class": "shift_axis",
            "from": "out_best",
            "axis": "T",
            "amount": 1,
            "pad_value": -1,  # to have always True at the first pos
        }
        # reinterpret time axis to work with following layers
        self.base_model.network["out_best_time_reinterpret"] = {
            "class": "reinterpret_data",
            "from": "out_best",
            "size_base": "shift_right",  # [B,T|shift_axis]
        }
        self.base_model.network["unique_mask"] = {
            "class": "compare",
            "kind": "not_equal",
            "from": ["out_best_time_reinterpret", "shift_right"],
        }
        self.base_model.network["non_blank_mask"] = {
            "class": "compare",
            "from": "out_best_time_reinterpret",
            "value": 10025,
            "kind": "not_equal",
        }
        self.base_model.network["out_best_mask"] = {
            "class": "combine",
            "kind": "logical_and",
            "from": ["unique_mask", "non_blank_mask"],
        }
        self.base_model.network["out_best_wo_blank"] = {
            "class": "masked_computation",
            "from": "out_best_time_reinterpret",
            "mask": "out_best_mask",
            "unit": {"class": "copy"},
            "target": "bpe_labels",
        }
        self.base_model.network["edit_distance"] = {
            "class": "copy",
            "from": "out_best_wo_blank",
            "only_on_search": True,
            "loss": "edit_distance",
            "target": "bpe_labels",
        }

        self.decision_layer_name = "out_best_wo_blank"

        return self.dec_output
