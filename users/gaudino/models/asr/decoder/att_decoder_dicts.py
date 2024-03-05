
def get_attention_decoder_dict(label_dim=10025, target_embed_dim=640):
    attention_decoder_dict = {
        # reinterpreted for target_embed
        "output_reinterpret": {
            "class": "reinterpret_data",
            "from": "output",
            "set_sparse": True,
            "set_sparse_dim": label_dim,  # V
            "initial_output": 0,
        },
        "prev_output_reinterpret": {
            "class": "copy",
            "from": "prev:output_reinterpret",
        },
        "trigg_att": {
            "class": "subnetwork",
            "from": [],
            "n_out": label_dim,
            "name_scope": "",
            "subnetwork": {
                "target_embed0": {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": "base:output_reinterpret",
                    "n_out": target_embed_dim,
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
        # "att_log_scores": {
        #     "class": "activation",
        #     "activation": "safe_log",
        #     "from": "trigg_att",
        # },
    }
    return attention_decoder_dict