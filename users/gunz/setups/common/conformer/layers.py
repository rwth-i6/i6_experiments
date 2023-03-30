__all__ = [
    "add_conv_layer",
    "add_pool_layer",
    "add_blstm_layer",
    "trafo_enc_layer",
    "trafo_dec_layer",
    "separated_trafo_enc_layer",
    "separated_trafo_ca_layer",
    "limited_window_trafo_enc_layer",
    "extended_trafo_enc_layer",
    "add_se_layer",
    "depthwise_separable_conv_layer",
    "add_conformer_block",
]

DEFAULT_INIT = "variance_scaling_initializer(mode='fan_in', distribution='uniform', scale=0.78)"


def add_conv_layer(
    network,
    idx,
    filter_size,
    padding,
    strides,
    dim,
    from_layers,
    *,
    with_bias=None,
    activation=None,
):
    """
    :param dict network: RETURNN network
    :param int idx: layer number
    :param tuple[int] filter_size: (width,), (height, width), (depth, height, width) ## 1D, 2D or 3D conv.
    :param str padding: "same" or "valid"
    :param  int|tuple[int] strides: should be same as filter_size ot int
    :param int dim: # output features
    :param None|list[str] from_layers: list of input layer names
    :param bool with_bias:
    :param None|str activation: activation function at end
    :return dict: RETURNN network
    """
    ## set the name of the conv layer with the given idx
    name = f"conv_{idx}"
    network[name] = {
        "class": "conv",
        "filter_size": filter_size,
        "padding": padding,
        "strides": strides,
        "n_out": dim,
        "from": from_layers,
    }
    if with_bias is not None:
        network[name]["with_bias"] = with_bias
    if activation is not None:
        network[name]["activation"] = activation

    if from_layers is None:
        del network[name]["from"]

    return network


def add_pool_layer(
    network,
    idx,
    mode,
    pool_size,
    padding,
    strides,
    from_layers,
    *,
    trainable=False,
    prefix=None,
):

    name = f"pool_{idx}" if not prefix else f"{prefix}pool_{idx}"

    network[name] = {
        "class": "pool",
        "mode": mode,
        "pool_size": pool_size,
        "padding": padding,
        "strides": strides,
        "from": from_layers,
        "trainable": trainable,
    }

    if from_layers is None:
        del network["from"]

    return network


def add_blstm_layer(
    network,
    idx,
    dim,
    dropout,
    l2,
    from_layers,
    *,
    unit="nativelstm2",
    max_seq_len=None,
    prefix=None,
):
    identifier = [
        [f"lstm_{idx}_fwd" if not prefix else f"{prefix}_lstm_{idx}_fwd", 1],
        [f"lstm_{idx}_bwd" if not prefix else f"{prefix}_lstm_{idx}_bwd", -1],
    ]

    for name, direction in identifier:
        network[name] = {
            "class": "rec",
            "unit": unit,
            "direction": direction,
            "n_out": dim,
            "dropout": dropout,
            "L2": l2,
            "from": from_layers,
        }

        if max_seq_len is not None:
            network[name]["max_seq_len"] = max_seq_len

        if from_layers is None:
            del network[name]["from"]

    return network


# self attention key total dim = key_dim
# self attention value total dim = value_dim
# feed forward dim = ff_dim

# sa_dropout = attention_dropout
# sa_post_dropout = postprocess_dropout
# ff_activation_dropout = act_dropout
# ff_post_dropout = postprocess_dropout

## used to build one transformer block consisting of layer normalization,
## multi-head attention followed by dropout and residual connection, and FFN sublayer followed by dropout and residual connenction
## the attention scores are also dropped out
## new par.: model_dim
def trafo_enc_layer(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
    attention_left_only=False,
):

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # self attention block
    ## first add a layer normalization
    network[f"{name}_self_att_laynorm"] = {"class": "layer_norm", "from": from_layers}
    ## one multi-head attention layer
    ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)
    network[f"{name}_self_att_att"] = {
        "class": "self_attention",
        "num_heads": num_heads,
        "total_key_dim": key_dim,
        "n_out": value_dim,
        "from": [f"{name}_self_att_laynorm"],
        "attention_left_only": attention_left_only,  ## masked self attention or not
        "attention_dropout": sa_dropout,
        "forward_weights_init": initialization,
    }
    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    # feed forward block
    ## two linear layers with activation in between
    ## ff_dim would be the size of hidden units
    ## the output of the FNN sub-layer would be input to the next transformer block and
    ## therefore the output dim. should be model_dim
    network[f"{name}_ff_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }
    network[f"{name}_ff_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff_conv2"],
    }
    network[f"{name}_ff_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_ff_drop"],
        "n_out": model_dim,  # value_dim,
    }
    network[f"{name}"] = {"class": "copy", "from": [f"{name}_ff_out"]}

    return network


def trafo_enc_layer_all_in_one(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
    end_layernorm=False,
    attention_left_only=False,
    separated=False,
    windowing=False,
    window_size=None,
    gauss_window=False,
    relative_pe=False,
    fixed=False,
    clipping=100,
):
    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # self attention block
    ## first add a layer normalization
    network[f"{name}_self_att_laynorm"] = {"class": "layer_norm", "from": from_layers}

    if separated:

        key_per_head = int(key_dim / num_heads)
        value_per_head = int(value_dim / num_heads)

        network[f"{name}_att_query0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": key_dim,
            "forward_weights_init": initialization,
        }

        # query per head
        network[f"{name}_att_query"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
            "from": [f"{name}_att_query0"],
        }

        network[f"{name}_att_key0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": key_dim,  # (B, enc-T, D)
            "forward_weights_init": initialization,
        }
        network[f"{name}_att_value0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": value_dim,
            "forward_weights_init": initialization,
        }

        ## split the key and value vectors for each head
        network[f"{name}_att_key"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, key_per_head),
            "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
        }

        network[f"{name}_att_value"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, value_per_head),
            "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
        }

        ## encoder-decoder energy
        ## we have exactly enc-T energy values
        network[f"{name}_att_energy"] = {
            "class": "dot",
            "red1": -1,
            "red2": -1,
            "var1": "T",
            "var2": "T?",
            "from": [f"{name}_att_key", f"{name}_att_query"],
        }  # (B, H, enc-T, enc-T) #(B, H, enc-T, 1)

        ## normalize the attention weights (depends on key/query dim.)
        network[f"{name}_att_weights"] = {
            "class": "softmax_over_spatial",
            "from": [f"{name}_att_energy"],
            "energy_factor": key_per_head**-0.5,  # (B, enc-T, H, 1)
        }

        ## attention weights dropout
        network[f"{name}_att_weights_drop"] = {
            "class": "dropout",
            "dropout_noise_shape": {"*": None},
            "dropout": sa_dropout,
            "from": [f"{name}_att_weights"],
        }

        ## now we have an attention weight value for each encoder-side output
        ## we get per head one vector
        network[f"{name}_att0"] = {
            "class": "generic_attention",
            "weights": f"{name}_att_weights_drop",
            "base": f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
        }

        network[f"{name}_self_att_att"] = {
            "class": "merge_dims",
            "axes": "static",  # "static"
            "from": [f"{name}_att0"],
        }

        ## not sure, if this works
        ## not sure, if this works
        if windowing:
            # hard masking
            if not gauss_window:
                eval_win_size = (
                    f"tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(tf.constant({window_size}, dtype=tf.int32), axis = -1), axis = -1), "
                    f"[1, tf.shape(source(0))[-2], tf.shape(source(0))[-1]]), 0)"
                )
                eval_win_start = (
                    f"tf.expand_dims(tf.map_fn(fn = lambda t: tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-1]), 0), "
                    f"[tf.shape(source(0))[2], 1]) - t, elems=tf.constant({window_size}, dtype=tf.int32)//2), 0)"
                )

                # eval_encoderT_pos = 'tf.tile(tf.expand_dims(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), '\
                #   '[1, tf.shape(source(0))[-1]]), 0), 0), [1, tf.shape(source(0))[1], 1, 1])'

                eval_encoderT_pos = (
                    "tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), "
                    "[tf.shape(source(0))[1], tf.shape(source(0))[-1]]), tf.shape(source(0))[1:]), 0)"
                )

                # without batch dim.
                # eval_masking = 'tf.logical_and(tf.less_equal(source(0), source(1)), tf.greater_equal(source(0), source(2)))'
                eval_masking = (
                    "tf.tile(tf.logical_and(tf.less_equal(source(0), source(1)), tf.greater_equal(source(0), source(2))), "
                    "[tf.shape(source(3))[0], 1, 1, 1])"
                )

                network[f"{name}_att_energy"]["out_type"] = {"time_dim_axis": 3}
                network[f"{name}_win_size"] = {
                    "class": "eval",
                    "eval": eval_win_size,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "int32"},
                }

                network[f"{name}_win_start"] = {
                    "class": "eval",
                    "eval": eval_win_start,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "int32"},
                }

                ## normalize the attention weights (depends on key/query dim.)
                # network[f"{name}_att_weights"]['window_start'] = f"{name}_win_start"
                # network[f"{name}_att_weights"]['window_size'] = f"{name}_win_size"

                network[f"{name}_win_end"] = {
                    "class": "combine",
                    "from": [f"{name}_win_start", f"{name}_win_size"],
                    "kind": "add",
                }

                network[f"{name}_encoderT_pos"] = {
                    "class": "eval",
                    "eval": eval_encoderT_pos,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "int32"},
                }

                network[f"{name}_masking"] = {
                    "class": "eval",
                    "eval": eval_masking,
                    "from": [
                        f"{name}_encoderT_pos",
                        f"{name}_win_end",
                        f"{name}_win_start",
                        f"{name}_att_energy",
                    ],
                    "out_type": {"dtype": "bool"},
                }

                network[f"{name}_att_energy_masked"] = {
                    "class": "eval",
                    "eval": f"tf.where(source(0), source(1), "
                    f"tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant(float('-inf')), 0), 0), 0), 0), tf.shape(source(1))))",
                    "from": [f"{name}_masking", f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }
            # soft masking: Gaussian window
            else:
                eval_key_pos = (
                    "tf.cast(tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), "
                    '[tf.shape(source(0))[1], tf.shape(source(0))[-1]]), tf.shape(source(0))[1:]), 0), "float32")'
                )
                eval_query_pos = (
                    f"tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-1]), 0), "
                    f'[tf.shape(source(0))[-2], 1]), 0), [{num_heads}, 1, 1]), 0), "float32")'
                )

                network[f"{name}_key_pos"] = {
                    "class": "eval",
                    "eval": eval_key_pos,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }
                network[f"{name}_query_pos"] = {
                    "class": "eval",
                    "eval": eval_query_pos,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }
                network[f"{name}_std_for_gaussian_window"] = {
                    "class": "variable",
                    "init": window_size[0],
                    "shape": (num_heads,),
                }

                network[f"{name}_masking"] = {
                    "class": "eval",
                    "eval": f"-0.5 * tf.square(source(0) - source(1)) / tf.reshape(tf.square(source(2)), [tf.shape(source(3))[0], {num_heads}, 1, 1])",
                    "from": [
                        f"{name}_query_pos",
                        f"{name}_key_pos",
                        f"{name}_std_for_gaussian_window",
                        f"{name}_att_energy",
                    ],
                    "out_type": {"dtype": "float32"},
                }

                network[f"{name}_att_energy_masked"] = {
                    "class": "combine",
                    "kind": "add",
                    "from": [f"{name}_masking", f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }

            network[f"{name}_att_weights"]["from"] = [f"{name}_att_energy_masked"]
            network[f"{name}_att_weights"]["use_time_mask"] = False

    else:
        ## one multi-head attention layer
        ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)
        network[f"{name}_self_att_att"] = {
            "class": "self_attention",
            "num_heads": num_heads,
            "total_key_dim": key_dim,
            "n_out": value_dim,
            "from": [f"{name}_self_att_laynorm"],
            "attention_left_only": attention_left_only,  ## masked self attention or not
            "attention_dropout": sa_dropout,
            "forward_weights_init": initialization,
        }
        if relative_pe:
            network[f"{name}_rel_pos"] = {
                "class": "relative_positional_encoding",
                "from": [f"{name}_self_att_laynorm"],
                "fixed": fixed,
                "clipping": clipping,
                "n_out": key_dim // num_heads,
                "forward_weights_init": initialization,
            }

            network[f"{name}_self_att_att"]["key_shift"] = f"{name}_rel_pos"

    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    # feed forward block
    ## two linear layers with activation in between
    ## ff_dim would be the size of hidden units
    ## the output of the FNN sub-layer would be input to the next transformer block and
    ## therefore the output dim. should be model_dim
    network[f"{name}_ff_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }
    network[f"{name}_ff_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff_conv2"],
    }
    network[f"{name}_ff_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_ff_drop"],
        "n_out": model_dim,  # value_dim,
    }
    if end_layernorm:
        network[f"{name}"] = {"class": "layer_norm", "from": [f"{name}_ff_out"]}
    else:
        network[f"{name}"] = {"class": "copy", "from": [f"{name}_ff_out"]}

    return network


def limited_window_trafo_enc_layer(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    window_size,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
):
    key_per_head = int(key_dim / num_heads)
    value_per_head = int(value_dim / num_heads)

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # self attention block
    ## first add a layer normalization
    network[f"{name}_self_att_laynorm"] = {"class": "layer_norm", "from": from_layers}

    network[f"{name}_att_query0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,
        "forward_weights_init": initialization,
    }

    # query per head
    network[f"{name}_att_query"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
        "from": [f"{name}_att_query0"],
    }

    network[f"{name}_att_key0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,  # (B, enc-T, D)
        "forward_weights_init": initialization,
    }
    network[f"{name}_att_value0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": value_dim,
        "forward_weights_init": initialization,
    }

    ## split the key and value vectors for each head
    network[f"{name}_att_key"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),
        "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
    }

    network[f"{name}_att_value"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, value_per_head),
        "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
    }

    ## encoder-decoder energy
    ## we have exactly enc-T energy values
    network[f"{name}_att_energy"] = {
        "class": "dot",
        "red1": -1,
        "red2": -1,
        "var1": "T",
        "var2": "T?",
        "from": [f"{name}_att_key", f"{name}_att_query"],
    }  # (B, H, enc-T, enc-T) #(B, H, enc-T, 1)

    # f'tf.expand_dims(tf.expand_dims(tf.tile(tf.expand_dims(tf.constant({window_size}, dtype=tf.int32), axis = -1), [1, tf.shape(source(0))[-2]]), 0), -1)'
    network[f"{name}_win_size"] = {
        "class": "eval",
        "eval": f"tf.expand_dims(tf.expand_dims(tf.tile(tf.expand_dims(tf.constant({window_size}, dtype=tf.int32), axis = -1), [1, tf.shape(source(0))[-1]]), 0), -2)",
        "from": [f"{name}_att_energy"],
        "out_type": {"dtype": "int32"},
    }

    # f'tf.expand_dims(tf.expand_dims(tf.map_fn(fn = lambda t: tf.range(tf.shape(source(0))[-2]) - t, elems=tf.constant({window_size}, dtype=tf.int32)//2), 0), -1)'
    network[f"{name}_win_start"] = {
        "class": "eval",
        "eval": f"tf.expand_dims(tf.expand_dims(tf.map_fn(fn = lambda t: tf.range(tf.shape(source(0))[-1]) - t, elems=tf.constant({window_size}, dtype=tf.int32)//2), 0), -2)",
        "from": [f"{name}_att_energy"],
        "out_type": {"dtype": "int32"},
    }

    ## normalize the attention weights (depends on key/query dim.)
    network[f"{name}_att_weights"] = {
        "class": "softmax_over_spatial",
        "from": [f"{name}_att_energy"],
        "energy_factor": key_per_head**-0.5,  # (B, H, enc-T, 1)
        "window_start": f"{name}_win_start",
        "window_size": f"{name}_win_size",
    }

    ## attention weights dropout
    network[f"{name}_att_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "dropout": sa_dropout,
        "from": [f"{name}_att_weights"],
    }

    ## now we have an attention weight value for each encoder-side output
    ## we get per head one vector
    network[f"{name}_att0"] = {
        "class": "generic_attention",
        "weights": f"{name}_att_weights_drop",
        "base": f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
    }

    network[f"{name}_self_att_att"] = {
        "class": "merge_dims",
        "axes": "static",  # "static"
        "from": [f"{name}_att0"],
    }

    ## one multi-head attention layer
    ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)

    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    # feed forward block
    ## two linear layers with activation in between
    ## ff_dim would be the size of hidden units
    ## the output of the FNN sub-layer would be input to the next transformer block and
    ## therefore the output dim. should be model_dim
    network[f"{name}_ff_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }
    network[f"{name}_ff_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff_conv2"],
    }
    network[f"{name}_ff_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_ff_drop"],
        "n_out": model_dim,  # value_dim,
    }
    network[f"{name}"] = {"class": "copy", "from": [f"{name}_ff_out"]}

    return network


def separated_trafo_enc_layer(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
):
    key_per_head = int(key_dim / num_heads)
    value_per_head = int(value_dim / num_heads)

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # self attention block
    ## first add a layer normalization
    network[f"{name}_self_att_laynorm"] = {"class": "layer_norm", "from": from_layers}

    network[f"{name}_att_query0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,
        "forward_weights_init": initialization,
    }

    # query per head
    network[f"{name}_att_query"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
        "from": [f"{name}_att_query0"],
    }

    network[f"{name}_att_key0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,  # (B, enc-T, D)
        "forward_weights_init": initialization,
    }
    network[f"{name}_att_value0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": value_dim,
        "forward_weights_init": initialization,
    }

    ## split the key and value vectors for each head
    network[f"{name}_att_key"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),
        "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
    }

    network[f"{name}_att_value"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, value_per_head),
        "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
    }

    ## encoder-decoder energy
    ## we have exactly enc-T energy values
    network[f"{name}_att_energy"] = {
        "class": "dot",
        "red1": -1,
        "red2": -1,
        "var1": "T",
        "var2": "T?",
        "from": [f"{name}_att_key", f"{name}_att_query"],
    }  # (B, H, enc-T, enc-T) #(B, H, enc-T, 1)

    ## normalize the attention weights (depends on key/query dim.)
    network[f"{name}_att_weights"] = {
        "class": "softmax_over_spatial",
        "from": [f"{name}_att_energy"],
        "energy_factor": key_per_head**-0.5,  # (B, enc-T, H, 1)
    }

    ## attention weights dropout
    network[f"{name}_att_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "dropout": sa_dropout,
        "from": [f"{name}_att_weights"],
    }

    ## now we have an attention weight value for each encoder-side output
    ## we get per head one vector
    network[f"{name}_att0"] = {
        "class": "generic_attention",
        "weights": f"{name}_att_weights_drop",
        "base": f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
    }

    network[f"{name}_self_att_att"] = {
        "class": "merge_dims",
        "axes": "static",  # "static"
        "from": [f"{name}_att0"],
    }

    ## one multi-head attention layer
    ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)

    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    # feed forward block
    ## two linear layers with activation in between
    ## ff_dim would be the size of hidden units
    ## the output of the FNN sub-layer would be input to the next transformer block and
    ## therefore the output dim. should be model_dim
    network[f"{name}_ff_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }
    network[f"{name}_ff_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff_conv2"],
    }
    network[f"{name}_ff_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_ff_drop"],
        "n_out": model_dim,  # value_dim,
    }
    network[f"{name}"] = {"class": "copy", "from": [f"{name}_ff_out"]}

    return network


def extended_trafo_enc_layer(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
    ff_layers=1,
    shared_KV=False,
):
    def ff_layer(network, ff_name, ff_dim, ff_out_dim, ff_from_layer, ff_activation):

        # layer norm
        network[f"{ff_name}_laynorm"] = {"class": "layer_norm", "from": [ff_from_layer]}
        network[f"{ff_name}_conv1"] = {
            "class": "linear",
            "activation": ff_activation,
            "with_bias": True,
            "from": [f"{ff_name}_laynorm"],
            "n_out": ff_dim,
            "forward_weights_init": initialization,
        }
        network[f"{ff_name}_conv2"] = {
            "class": "linear",
            "activation": None,
            "with_bias": True,
            "from": [f"{ff_name}_conv1"],
            "dropout": ff_activation_dropout,
            "n_out": ff_out_dim,
            "forward_weights_init": initialization,
        }
        network[f"{ff_name}_drop"] = {
            "class": "dropout",
            "dropout": ff_post_dropout,
            "from": [f"{ff_name}_conv2"],
        }
        network[f"{ff_name}_out"] = {
            "class": "combine",
            "kind": "add",
            "from": [ff_from_layer, f"{ff_name}_drop"],
            "n_out": ff_out_dim,  # value_dim,
        }

    key_per_head = int(key_dim / num_heads)
    value_per_head = int(value_dim / num_heads)

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # self attention block
    ## first add a layer normalization
    network[f"{name}_self_att_laynorm"] = {"class": "layer_norm", "from": from_layers}

    network[f"{name}_att_query0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,
        "forward_weights_init": initialization,
    }

    # query per head
    network[f"{name}_att_query"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
        "from": [f"{name}_att_query0"],
    }

    network[f"{name}_att_key0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,  # (B, enc-T, D)
        "forward_weights_init": initialization,
    }

    ## split the key and value vectors for each head
    network[f"{name}_att_key"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),
        "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
    }

    # eventually share the KV projection matrices
    # since it's SA ==> identical key and value sequences
    if shared_KV:
        network[f"{name}_att_value0"] = {"class": "copy", "from": [f"{name}_att_key0"]}
        network[f"{name}_att_value"] = {"class": "copy", "from": [f"{name}_att_key"]}
    else:
        network[f"{name}_att_value0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": value_dim,
            "forward_weights_init": initialization,
        }
        network[f"{name}_att_value"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, value_per_head),
            "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
        }

    ## encoder-decoder energy
    ## we have exactly enc-T energy values
    network[f"{name}_att_energy"] = {
        "class": "dot",
        "red1": -1,
        "red2": -1,
        "var1": "T",
        "var2": "T?",
        "from": [f"{name}_att_key", f"{name}_att_query"],
    }  # (B, H, enc-T, enc-T) #(B, H, enc-T, 1)

    ## normalize the attention weights (depends on key/query dim.)
    network[f"{name}_att_weights"] = {
        "class": "softmax_over_spatial",
        "from": [f"{name}_att_energy"],
        "energy_factor": key_per_head**-0.5,  # (B, enc-T, H, 1)
    }

    ## attention weights dropout
    network[f"{name}_att_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "dropout": sa_dropout,
        "from": [f"{name}_att_weights"],
    }

    ## now we have an attention weight value for each encoder-side output
    ## we get per head one vector
    network[f"{name}_att0"] = {
        "class": "generic_attention",
        "weights": f"{name}_att_weights_drop",
        "base": f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
    }

    network[f"{name}_self_att_att"] = {
        "class": "merge_dims",
        "axes": "static",  # "static"
        "from": [f"{name}_att0"],
    }

    ## one multi-head attention layer
    ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)

    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    # feed forward block
    ## two linear layers with activation in between
    ## ff_dim would be the size of hidden units
    ## the output of the FNN sub-layer would be input to the next transformer block and
    ## therefore the output dim. should be model_dim

    for i in range(ff_layers):

        ff_name = f"{name}_ff{i:03d}"

        if i == 0:
            ff_from_layer = f"{name}_self_att_out"
        else:
            ff_from_layer = f"{name}_ff{(i - 1):03d}_out"

        ff_layer(network, ff_name, ff_dim, model_dim, ff_from_layer, ff_activation)

    network[f"{name}"] = {
        "class": "copy",
        "from": [f"{name}_ff{(ff_layers - 1):03d}_out"],
    }

    return network


def add_se_layer(
    network,
    se_name,
    se_from_layer,
    in_channels,
    out_channels=None,
    if_bias=True,
    se_activation="swish",
):

    if not out_channels:
        out_channels = in_channels // 8
    # assume se_from_layer has output shape (batch, time, feature)
    ## squeeze operation
    # reduce along the time axes
    network[f"{se_name}_squeeze"] = {
        "class": "reduce",
        "mode": "mean",
        "keep_dims": True,
        "from": [se_from_layer],
        "axes": "time",
    }

    ## excitation operation (batch, feature)
    network[f"{se_name}_excitation_layer1"] = {
        "class": "linear",
        "activation": se_activation,
        "with_bias": if_bias,
        "n_out": out_channels,
        "from": [f"{se_name}_squeeze"],
    }

    # (batch, feature)
    network[f"{se_name}_excitation_layer2"] = {
        "class": "linear",
        "activation": "sigmoid",
        "with_bias": if_bias,
        "n_out": in_channels,
        "from": [f"{se_name}_excitation_layer1"],
    }
    # Todo: a tile layer to tile out the time layer
    # element-wise multiplication
    network[f"{se_name}_output"] = {
        "class": "combine",
        "kind": "mul",
        "from": [f"{se_name}_excitation_layer2", se_from_layer],
    }
    return network


def depthwise_separable_conv_layer(
    separable_from_layer,
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    channel_multiplier=1,
):
    ## assume the data format is NH(expanded)W(time)C(feature)
    depthwise_filter_shape = [1, kernel_size, in_channels, float(channel_multiplier)]
    pointwise_filter_shape = [1, 1, channel_multiplier * in_channels, out_channels]

    strides = [1, 1, stride, 1]
    padding = "SAME"

    eval_str = f'tf.nn.separable_conv2d(source(0), depthwise_filter=tf.Variable(tf.glorot_uniform_initializer()(shape={depthwise_filter_shape})), pointwise_filter=tf.Variable(tf.glorot_uniform_initializer()(shape={pointwise_filter_shape})), strides={strides}, padding="{padding}")'

    return {"class": "eval", "from": [separable_from_layer], "eval": eval_str}


# the from_layer works as state

# the new information added to the state
# is always extracted as a weighted (the weighting factor from the dot-product + sigmoid) linear combination of the
# context vector
def add_context_layer_experimental(
    network,
    idx,
    from_layer,
    model_size,
    context_size=9,
    conv_activation="relu",
    linear_activation=None,
    initialization=DEFAULT_INIT,
    discard=False,
):

    network[f"enc_{idx:03d}_context"] = {
        "class": "conv",
        "from": from_layer,
        "padding": "same",
        "filter_size": (context_size,),
        "n_out": model_size,
        "activation": conv_activation,
        "with_bias": True,
    }

    # both have shape (B, T, model_dim//2)
    network[f"enc_{idx:03d}_excitation"] = {
        "class": "combine",
        "kind": "mul",
        "from": [from_layer, f"enc_{idx:03d}_context"],
    }
    # (B, T, 1)
    network[f"enc_{idx:03d}_excitation_score"] = {
        "class": "reduce",
        "mode": "sum",
        "from": [f"enc_{idx:03d}_excitation"],
        "axis": "F",
        "keep_dims": True,
    }
    # (B, T, 1)
    network[f"enc_{idx:03d}_excitation_weight"] = {
        "class": "activation",
        "activation": "sigmoid",
        "from": [f"enc_{idx:03d}_excitation_score"],
    }

    network[f"enc_{idx:03d}_context_transformed"] = {
        "class": "linear",
        "activation": linear_activation,
        "with_bias": True,
        "from": [f"enc_{idx:03d}_context"],
        "n_out": model_size,
        "forward_weights_init": initialization,
    }

    network[f"enc_{idx:03d}_state_new"] = {
        "class": "eval",
        "eval": "tf.add(source(0), tf.multiply(source(1), source(2)))"
        if not discard
        else "tf.add(tf.multiply(source(0), tf.subtract(1.0, source(2))), tf.multiply(source(1), source(2)))",
        "from": [
            from_layer,
            f"enc_{idx:03d}_context_transformed",
            f"enc_{idx:03d}_excitation_weight",
        ],
    }

    network[f"enc_{idx:03d}_self_att_out"] = {
        "class": "copy",
        "from": [f"enc_{idx:03d}_state_new"],
    }


def add_context_layer(
    network,
    idx,
    from_layer,
    model_size,
    context_size=9,
    state_ratio=1 / 2,
    conv_activation="relu",
    linear_activation=None,
    initialization=DEFAULT_INIT,
    depthwise_separable=False,
):

    state_size = int(model_size * state_ratio)

    # slice layer to do dot-product
    network[f"enc_{idx:03d}_state"] = {
        "class": "slice",
        "from": from_layer,
        "axis": 2,
        "slice_start": 0,
        "slice_end": state_size,
        "slice_step": 1,
    }

    network[f"enc_{idx:03d}_memory"] = {
        "class": "slice",
        "from": from_layer,
        "axis": 2,
        "slice_start": state_size,
        "slice_end": model_size,
        "slice_step": 1,
    }
    # maps (B, T, F) to (B, T, F'): (context_size,) * F * F' parameters
    # (context_size,) * F + F' * F parameters
    # maps (B, T, F) to (B, T, F) and then to (B, T, F')
    if not depthwise_separable:
        network[f"enc_{idx:03d}_context"] = {
            "class": "conv",
            "from": f"enc_{idx:03d}_state",
            "padding": "same",
            "filter_size": (context_size,),
            "n_out": model_size - state_size,  # memory size
            "activation": conv_activation,
            "with_bias": True,
        }
    else:
        network[f"enc_{idx:03d}_context_pre"] = {
            "class": "depthwise_convolution1D",
            "kernel_size": context_size,
            "from": [f"enc_{idx:03d}_state"],
        }

        network[f"enc_{idx:03d}_context"] = {
            "class": "linear",
            "activation": conv_activation,
            "with_bias": True,
            "n_out": model_size - state_size,
            "from": [f"enc_{idx:03d}_context_pre"],
            "forward_weights_init": initialization,
        }

    # both have shape (B, T, model_dim//2)
    network[f"enc_{idx:03d}_excitation"] = {
        "class": "combine",
        "kind": "mul",
        "from": [f"enc_{idx:03d}_memory", f"enc_{idx:03d}_context"],
    }
    # (B, T, 1)
    network[f"enc_{idx:03d}_excitation_score"] = {
        "class": "reduce",
        "mode": "sum",
        "from": [f"enc_{idx:03d}_excitation"],
        "axis": "F",
        "keep_dims": True,
    }
    # (B, T, 1)
    network[f"enc_{idx:03d}_excitation_weight"] = {
        "class": "activation",
        "activation": "sigmoid",
        "from": [f"enc_{idx:03d}_excitation_score"],
    }

    # first part old + weight * context
    # second part (1-weight)*old + weight * context
    network[f"enc_{idx:03d}_memory_new"] = {
        "class": "eval",
        "eval": "tf.add(tf.multiply(source(0), tf.subtract(1.0, source(2))), tf.multiply(source(1), source(2)))",
        "from": [
            f"enc_{idx:03d}_memory",
            f"enc_{idx:03d}_context",
            f"enc_{idx:03d}_excitation_weight",
        ],
    }
    network[f"enc_{idx:03d}_context_transformed"] = {
        "class": "linear",
        "activation": linear_activation,
        "with_bias": True,
        "from": [f"enc_{idx:03d}_context"],
        "n_out": state_size,
        "forward_weights_init": initialization,
    }

    network[f"enc_{idx:03d}_state_new"] = {
        "class": "eval",
        "eval": "tf.add(source(0), tf.multiply(source(1), source(2)))",
        "from": [
            f"enc_{idx:03d}_state",
            f"enc_{idx:03d}_context_transformed",
            f"enc_{idx:03d}_excitation_weight",
        ],
    }

    network[f"enc_{idx:03d}_self_att_out"] = {
        "class": "copy",
        "from": [f"enc_{idx:03d}_state_new", f"enc_{idx:03d}_memory_new"],
    }


def conformer_enc_layer_all_in_one(*args, tbs_version="default", **kwargs):
    assert tbs_version == "default"

    return conformer_enc_layer_all_in_one_default(*args, **kwargs)


# untied_pe=False, relative_pe_transformer_xl=False
# energy_factor
def conformer_enc_layer_all_in_one_default(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    kernel_size,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    conv_post_dropout,
    initialization=DEFAULT_INIT,
    ff_activation="swish",
    end_layernorm=False,
    normal_conv=False,
    output_channels=16,
    kernel_size_for_feature=3,
    attention_left_only=False,
    separated=False,
    windowing=False,
    window_size=None,
    gauss_window=False,
    relative_pe=False,
    fixed=False,
    clipping=100,
    untied_pe=False,
    relative_pe_transformer_xl=False,
    linear_mapping=True,
    linear_mapping_bias=False,
    switch=False,
    energy_factor=-0.5,
    layer_norm_instead_of_batch_norm=False,
):
    if windowing or untied_pe or relative_pe_transformer_xl or energy_factor != -0.5:
        assert separated

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    norm_in_between = "layer_norm" if layer_norm_instead_of_batch_norm else "batch_norm"

    ## first ffn with residual connection
    network[f"{name}_ff1_laynorm"] = {"class": "layer_norm", "from": from_layers}
    network[f"{name}_ff1_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff1_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_ff1_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff1_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff1_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff1_conv2"],
    }

    network[f"{name}_ff1_drop_half"] = {
        "class": "eval",
        "eval": f"0.5 * source(0)",
        "from": [f"{name}_ff1_drop"],
    }
    network[f"{name}_ff1_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_ff1_drop_half"],
    }

    ## MHSA module
    network[f"{name}_self_att_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_ff1_out"],
    }

    if separated:
        key_per_head = int(key_dim / num_heads)
        value_per_head = int(value_dim / num_heads)

        network[f"{name}_att_query0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": key_dim,
            "forward_weights_init": initialization,
        }

        # query per head
        network[f"{name}_att_query"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
            "from": [f"{name}_att_query0"],
        }

        network[f"{name}_att_key0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": key_dim,  # (B, enc-T, D)
            "forward_weights_init": initialization,
        }
        network[f"{name}_att_value0"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "from": [f"{name}_self_att_laynorm"],
            "n_out": value_dim,
            "forward_weights_init": initialization,
        }

        ## split the key and value vectors for each head
        network[f"{name}_att_key"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, key_per_head),
            "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
        }

        network[f"{name}_att_value"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (num_heads, value_per_head),
            "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
        }

        ## encoder-decoder energy
        ## we have exactly enc-T energy values
        network[f"{name}_att_energy"] = {
            "class": "dot",
            "red1": -1,
            "red2": -1,
            "var1": "T",
            "var2": "T?",
            "from": [f"{name}_att_key", f"{name}_att_query"],
        }  # (B, H, key-T, query-T)

        ## normalize the attention weights (depends on key/query dim.)
        network[f"{name}_att_weights"] = {
            "class": "softmax_over_spatial",
            "from": [f"{name}_att_energy"],
            "energy_factor": key_per_head
            ** energy_factor,  # (B, H, key-T, query-T), key-T is where softmax is performed
        }

        # relative_pe as in transformer xl
        if relative_pe_transformer_xl and not relative_pe and not untied_pe:

            shared_layers = False
            network[f"{name}_att_emb_emb"] = network[f"{name}_att_energy"]

            # (B, enc-T, d_pos)
            assert "source" in network
            if "pos" not in network:
                network["pos"] = {
                    "class": "positional_encoding",
                    "add_to_input": False,
                    "from": ["source"],
                    "n_out": model_dim,
                }
            # network['pos_with_0'] = {
            #   "class": "eval", "from": ["pos"],
            #   "eval": f"tf.slice(tf.concat([tf.expand_dims(tf.tile(tf.reshape([0, 1] * ({model_dim}//2), " \
            #           f"(1, {model_dim})), [tf.shape(source(0))[0], 1]), 1), source(0)], 1), [0, 0, 0], [-1, tf.shape(source(0))[1], -1])"}

            if shared_layers:
                network["att_pos_key0"] = {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["pos"],
                    "n_out": key_dim,  # (B, enc-T, D) # pos_with_0
                    "forward_weights_init": initialization,
                }
                network["att_pos_key"] = {
                    "class": "split_dims",
                    "axis": "F",
                    "dims": (num_heads, key_per_head),
                    "from": ["att_pos_key0"],  # (B, enc-T, H, D/H)
                }
            else:
                network[f"{name}_att_pos_key0"] = {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["pos"],
                    "n_out": key_dim,  # (B, enc-T, D) # pos_with_0
                    "forward_weights_init": initialization,
                }
                network[f"{name}_att_pos_key"] = {
                    "class": "split_dims",
                    "axis": "F",
                    "dims": (num_heads, key_per_head),
                    "from": [f"{name}_att_pos_key0"],  # (B, enc-T, H, D/H)
                }

            # (B, enc-T, H, D/H), (B, dec-T, H, D/H) -> (B, H, enc-T, dec-T)
            network[f"{name}_att_emb_pos"] = {
                "class": "dot",
                "red1": -1,
                "red2": -1,
                "var1": "T",
                "var2": "T?",
                "from": [f"{name}_att_pos_key", f"{name}_att_query"],
            }

            if shared_layers:
                network[f"{name}_att_emb_pos"]["from"] = [
                    "att_pos_key",
                    f"{name}_att_query",
                ]

            # (B, H, enc-T, dec-T)
            network[f"{name}_att_emb_pos_shifted"] = {
                "class": "eval",
                "eval": "self.network.get_config().typed_value('rel_shift')(source(0))",
                "from": [f"{name}_att_emb_pos"],
                "out_type": {
                    "shape": (num_heads, None, None),
                    "batch_dim_axis": 0,
                    "time_dim_axis": 2,
                    "feature_dim_axis": 1,
                },
            }

            # (B, 4, F)
            if shared_layers:
                network["pos_emb_bias"] = {
                    "class": "variable",
                    "shape": (num_heads, key_per_head),
                    "add_time_axis": True,
                    "init": DEFAULT_INIT,
                }
            else:
                network[f"{name}_pos_emb_bias"] = {
                    "class": "variable",
                    "shape": (num_heads, key_per_head),
                    "add_time_axis": True,
                    "init": DEFAULT_INIT,
                }
            # (B, enc-T, H, D / H), (B, 1, H, D / H) --> (B, H, enc-T, dec-T=1)
            network[f"{name}_att_pos_emb"] = {
                "class": "dot",
                "red1": -1,
                "red2": -1,
                "var1": "T",
                "var2": "T?",
                "from": [f"{name}_att_key", f"{name}_pos_emb_bias"],
                "out_type": {"shape": (num_heads, None, 1)}
                #'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, "dim": num_heads}
            }

            if shared_layers:
                network[f"{name}_att_pos_emb"]["from"] = [
                    f"{name}_att_key",
                    "pos_emb_bias",
                ]

            network[f"{name}_att_pos_emb_tiled"] = {
                "class": "rel_shift",
                "rel_shift": False,
                "from": [f"{name}_att_pos_emb"],
                "out_type": {
                    "shape": (num_heads, None, None),
                    "batch_dim_axis": 0,
                    "time_dim_axis": 2,
                    "feature_dim_axis": 1,
                    "dim": num_heads,
                },
            }
            if shared_layers:
                network["pos_pos_bias"] = {
                    "class": "variable",
                    "shape": (num_heads, key_per_head),  # (B, d, 4)
                    "add_time_axis": True,
                    "init": DEFAULT_INIT,
                }

                # (B, enc - T, H, D / H), (B, 1, H, D / H) --> (B, H, enc-T, dec-T = 1)
                network["att_pos_pos"] = {
                    "class": "dot",
                    "red1": -1,
                    "red2": -1,
                    "var1": "T",
                    "var2": "T?",
                    "from": ["att_pos_key", "pos_pos_bias"],
                    "out_type": {"shape": (num_heads, None, 1)}
                    # 'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, "dim": num_heads}
                }

                # (B, H, T, T')
                network["att_pos_pos_shifted"] = {
                    "class": "rel_shift",
                    "from": ["att_pos_pos"],
                    "out_type": {
                        "shape": (num_heads, None, None),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "dim": num_heads,
                    },
                }
            else:
                network[f"{name}_pos_pos_bias"] = {
                    "class": "variable",
                    "shape": (num_heads, key_per_head),  # (B, d, 4)
                    "add_time_axis": True,
                    "init": DEFAULT_INIT,
                }

                # (B, enc - T, H, D / H), (B, 1, H, D / H) --> (B, H, enc-T, dec-T = 1)
                network[f"{name}_att_pos_pos"] = {
                    "class": "dot",
                    "red1": -1,
                    "red2": -1,
                    "var1": "T",
                    "var2": "T?",
                    "from": [f"{name}_att_pos_key", f"{name}_pos_pos_bias"],
                    "out_type": {"shape": (num_heads, None, 1)}
                    #'batch_dim_axis': 0, 'time_dim_axis': 2, "feature_dim_axis": 1, "dim": num_heads}
                }

                # (B, H, T, T')
                network[f"{name}_att_pos_pos_shifted"] = {
                    "class": "rel_shift",
                    "from": [f"{name}_att_pos_pos"],
                    "out_type": {
                        "shape": (num_heads, None, None),
                        "batch_dim_axis": 0,
                        "time_dim_axis": 2,
                        "feature_dim_axis": 1,
                        "dim": num_heads,
                    },
                }

            network[f"{name}_att_energy"] = {
                "class": "combine",
                "kind": "add",
                "from": [
                    f"{name}_att_emb_emb",
                    f"{name}_att_pos_emb_tiled",
                    f"{name}_att_emb_pos_shifted",
                    f"{name}_att_pos_pos_shifted",
                ],
            }
            if shared_layers:
                network[f"{name}_att_energy"]["from"] = [
                    f"{name}_att_emb_emb",
                    f"{name}_att_pos_emb_tiled",
                    f"{name}_att_emb_pos_shifted",
                    "att_pos_pos_shifted",
                ]

        if untied_pe and not relative_pe:
            assert "source" in network
            if "pos" not in network:
                network["pos"] = {
                    "class": "positional_encoding",
                    "add_to_input": False,
                    "from": ["source"],
                    "n_out": model_dim,
                }
            # shared
            if False:
                if "att_pos_query0" not in network:
                    network["att_pos_query0"] = {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["pos"],
                        "n_out": key_dim,
                        "forward_weights_init": initialization,
                    }

                    network["att_pos_query"] = {
                        "class": "split_dims",
                        "axis": "F",
                        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
                        "from": ["att_pos_query0"],
                    }

                    network["att_pos_key0"] = {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["pos"],
                        "n_out": key_dim,  # (B, enc-T, D)
                        "forward_weights_init": initialization,
                    }
                    network["att_pos_key"] = {
                        "class": "split_dims",
                        "axis": "F",
                        "dims": (num_heads, key_per_head),
                        "from": ["att_pos_key0"],  # (B, enc-T, H, D/H)
                    }

                    network["att_pos_energy"] = {
                        "class": "dot",
                        "red1": -1,
                        "red2": -1,
                        "var1": "T",
                        "var2": "T?",
                        "from": ["att_pos_key", "att_pos_query"],
                    }

                network[f"{name}_att_energy_with_pos_corr"] = {
                    "class": "combine",
                    "kind": "add",
                    "from": [f"{name}_att_energy", "att_pos_energy"],
                }

            # per layer
            if False:
                network[f"{name}_att_pos_query0"] = {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["pos"],
                    "n_out": key_dim,
                    "forward_weights_init": initialization,
                }

                network[f"{name}_att_pos_query"] = {
                    "class": "split_dims",
                    "axis": "F",
                    "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
                    "from": [f"{name}_att_pos_query0"],
                }

                network[f"{name}_att_pos_key0"] = {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["pos"],
                    "n_out": key_dim,  # (B, enc-T, D)
                    "forward_weights_init": initialization,
                }
                network[f"{name}_att_pos_key"] = {
                    "class": "split_dims",
                    "axis": "F",
                    "dims": (num_heads, key_per_head),
                    "from": [f"{name}_att_pos_key0"],  # (B, enc-T, H, D/H)
                }

                network[f"{name}_att_pos_energy"] = {
                    "class": "dot",
                    "red1": -1,
                    "red2": -1,
                    "var1": "T",
                    "var2": "T?",
                    "from": [f"{name}_att_pos_key", f"{name}_att_pos_query"],
                }

                network[f"{name}_att_energy_with_pos_corr"] = {
                    "class": "combine",
                    "kind": "add",
                    "from": [f"{name}_att_energy", f"{name}_att_pos_energy"],
                }

            # with corrected normalization factor
            if True:
                network[f"{name}_att_pos_query0"] = {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["pos"],
                    "n_out": key_dim,
                    "forward_weights_init": initialization,
                }

                network[f"{name}_att_pos_query"] = {
                    "class": "split_dims",
                    "axis": "F",
                    "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
                    "from": [f"{name}_att_pos_query0"],
                }

                network[f"{name}_att_pos_key0"] = {
                    "class": "linear",
                    "activation": None,
                    "with_bias": False,
                    "from": ["pos"],
                    "n_out": key_dim,  # (B, enc-T, D)
                    "forward_weights_init": initialization,
                }
                network[f"{name}_att_pos_key"] = {
                    "class": "split_dims",
                    "axis": "F",
                    "dims": (num_heads, key_per_head),
                    "from": [f"{name}_att_pos_key0"],  # (B, enc-T, H, D/H)
                }

                network[f"{name}_att_pos_energy"] = {
                    "class": "dot",
                    "red1": -1,
                    "red2": -1,
                    "var1": "T",
                    "var2": "T?",
                    "from": [f"{name}_att_pos_key", f"{name}_att_pos_query"],
                }

                network[f"{name}_att_energy_with_pos_corr"] = {
                    "class": "combine",
                    "kind": "add",
                    "from": [f"{name}_att_energy", f"{name}_att_pos_energy"],
                }

                network[f"{name}_att_weights"]["energy_factor"] = (2 * key_per_head) ** energy_factor

            # scale per layer
            if False:
                if "att_pos_query0" not in network:
                    network["att_pos_query0"] = {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["pos"],
                        "n_out": key_dim,
                        "forward_weights_init": initialization,
                    }

                    network["att_pos_query"] = {
                        "class": "split_dims",
                        "axis": "F",
                        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
                        "from": ["att_pos_query0"],
                    }

                    network["att_pos_key0"] = {
                        "class": "linear",
                        "activation": None,
                        "with_bias": False,
                        "from": ["pos"],
                        "n_out": key_dim,  # (B, enc-T, D)
                        "forward_weights_init": initialization,
                    }
                    network["att_pos_key"] = {
                        "class": "split_dims",
                        "axis": "F",
                        "dims": (num_heads, key_per_head),
                        "from": ["att_pos_key0"],  # (B, enc-T, H, D/H)
                    }

                    network["att_pos_energy"] = {
                        "class": "dot",
                        "red1": -1,
                        "red2": -1,
                        "var1": "T",
                        "var2": "T?",
                        "from": ["att_pos_key", "att_pos_query"],
                    }

                network[f"{name}_att_pos_energy_scale"] = {
                    "class": "variable",
                    "shape": (num_heads,),
                    "init": 1.0,
                    "add_batch_axis": False,
                }
                network[f"{name}_att_energy_with_pos_corr"] = {
                    "class": "eval",
                    "eval": f"tf.add(source(0), tf.multiply(source(1), tf.reshape(source(2), (1, {num_heads}, 1, 1))))",
                    "from": [
                        f"{name}_att_energy",
                        "att_pos_energy",
                        f"{name}_att_pos_energy_scale",
                    ],
                }

            network[f"{name}_att_weights"]["from"] = [f"{name}_att_energy_with_pos_corr"]

        ## attention weights dropout
        network[f"{name}_att_weights_drop"] = {
            "class": "dropout",
            "dropout_noise_shape": {"*": None},
            "dropout": sa_dropout,
            "from": [f"{name}_att_weights"],
        }

        ## now we have an attention weight value for each encoder-side output
        ## we get per head one vector
        network[f"{name}_att0"] = {
            "class": "generic_attention",
            "weights": f"{name}_att_weights_drop",
            "base": f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
        }

        network[f"{name}_self_att_att"] = {
            "class": "merge_dims",
            "axes": "static",  # "static"
            "from": [f"{name}_att0"],
        }

        ## not sure, if this works
        if windowing:
            # hard masking
            if not gauss_window:
                eval_win_size = (
                    f"tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(tf.constant({window_size}, dtype=tf.int32), axis = -1), axis = -1), "
                    f"[1, tf.shape(source(0))[-2], tf.shape(source(0))[-1]]), 0)"
                )
                eval_win_start = (
                    f"tf.expand_dims(tf.map_fn(fn = lambda t: tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-1]), 0), "
                    f"[tf.shape(source(0))[2], 1]) - t, elems=tf.constant({window_size}, dtype=tf.int32)//2), 0)"
                )

                # eval_encoderT_pos = 'tf.tile(tf.expand_dims(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), '\
                #   '[1, tf.shape(source(0))[-1]]), 0), 0), [1, tf.shape(source(0))[1], 1, 1])'

                eval_encoderT_pos = (
                    "tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), "
                    "[tf.shape(source(0))[1], tf.shape(source(0))[-1]]), tf.shape(source(0))[1:]), 0)"
                )

                # without batch dim.
                # eval_masking = 'tf.logical_and(tf.less_equal(source(0), source(1)), tf.greater_equal(source(0), source(2)))'
                eval_masking = (
                    "tf.tile(tf.logical_and(tf.less_equal(source(0), source(1)), tf.greater_equal(source(0), source(2))), "
                    "[tf.shape(source(3))[0], 1, 1, 1])"
                )

                network[f"{name}_att_energy"]["out_type"] = {"time_dim_axis": 3}
                network[f"{name}_win_size"] = {
                    "class": "eval",
                    "eval": eval_win_size,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "int32"},
                }

                network[f"{name}_win_start"] = {
                    "class": "eval",
                    "eval": eval_win_start,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "int32"},
                }

                ## normalize the attention weights (depends on key/query dim.)
                # network[f"{name}_att_weights"]['window_start'] = f"{name}_win_start"
                # network[f"{name}_att_weights"]['window_size'] = f"{name}_win_size"

                network[f"{name}_win_end"] = {
                    "class": "combine",
                    "from": [f"{name}_win_start", f"{name}_win_size"],
                    "kind": "add",
                }

                network[f"{name}_encoderT_pos"] = {
                    "class": "eval",
                    "eval": eval_encoderT_pos,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "int32"},
                }

                network[f"{name}_masking"] = {
                    "class": "eval",
                    "eval": eval_masking,
                    "from": [
                        f"{name}_encoderT_pos",
                        f"{name}_win_end",
                        f"{name}_win_start",
                        f"{name}_att_energy",
                    ],
                    "out_type": {"dtype": "bool"},
                }

                network[f"{name}_att_energy_masked"] = {
                    "class": "eval",
                    "eval": f"tf.where(source(0), source(1), "
                    f"tf.tile(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant(float('-inf')), 0), 0), 0), 0), tf.shape(source(1))))",
                    "from": [f"{name}_masking", f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }
            # soft masking: Gaussian window
            else:
                eval_key_pos = (
                    "tf.cast(tf.expand_dims(tf.reshape(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-2]), -1), "
                    '[tf.shape(source(0))[1], tf.shape(source(0))[-1]]), tf.shape(source(0))[1:]), 0), "float32")'
                )
                eval_query_pos = (
                    f"tf.cast(tf.expand_dims(tf.tile(tf.expand_dims(tf.tile(tf.expand_dims(tf.range(tf.shape(source(0))[-1]), 0), "
                    f'[tf.shape(source(0))[-2], 1]), 0), [{num_heads}, 1, 1]), 0), "float32")'
                )

                network[f"{name}_key_pos"] = {
                    "class": "eval",
                    "eval": eval_key_pos,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }
                network[f"{name}_query_pos"] = {
                    "class": "eval",
                    "eval": eval_query_pos,
                    "from": [f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }
                network[f"{name}_std_for_gaussian_window"] = {
                    "class": "variable",
                    "init": window_size[0],
                    "shape": (num_heads,),
                }

                network[f"{name}_masking"] = {
                    "class": "eval",
                    "eval": f"-0.5 * tf.square(source(0) - source(1)) / tf.reshape(tf.square(source(2)), [tf.shape(source(3))[0], {num_heads}, 1, 1])",
                    "from": [
                        f"{name}_query_pos",
                        f"{name}_key_pos",
                        f"{name}_std_for_gaussian_window",
                        f"{name}_att_energy",
                    ],
                    "out_type": {"dtype": "float32"},
                }

                network[f"{name}_att_energy_masked"] = {
                    "class": "combine",
                    "kind": "add",
                    "from": [f"{name}_masking", f"{name}_att_energy"],
                    "out_type": {"dtype": "float32"},
                }

            network[f"{name}_att_weights"]["from"] = [f"{name}_att_energy_masked"]
            network[f"{name}_att_weights"]["use_time_mask"] = False

    else:
        network[f"{name}_self_att_att"] = {
            "class": "self_attention",
            "num_heads": num_heads,
            "total_key_dim": key_dim,
            "n_out": value_dim,
            "from": [f"{name}_self_att_laynorm"],
            "attention_left_only": attention_left_only,
            "attention_dropout": sa_dropout,
            "forward_weights_init": initialization,
        }

        if relative_pe:
            network[f"{name}_rel_pos"] = {
                "class": "relative_positional_encoding",
                "from": [f"{name}_self_att_laynorm"],
                "fixed": fixed,
                "clipping": clipping,
                "n_out": key_dim // num_heads,
                "forward_weights_init": initialization,
            }
            network[f"{name}_self_att_att"]["key_shift"] = f"{name}_rel_pos"
    if linear_mapping:
        network[f"{name}_self_att_lin"] = {
            "class": "linear",
            "activation": None,
            "with_bias": linear_mapping_bias,
            "from": [f"{name}_self_att_att"],
            "n_out": model_dim,
            "forward_weights_init": initialization,
        }
        network[f"{name}_self_att_drop"] = {
            "class": "dropout",
            "dropout": sa_post_dropout,
            "from": [f"{name}_self_att_lin"],
        }
    else:
        network[f"{name}_self_att_drop"] = {
            "class": "dropout",
            "dropout": sa_post_dropout,
            "from": [f"{name}_self_att_att"],
        }

    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_ff1_out", f"{name}_self_att_drop"],
        "n_out": model_dim,
    }
    ## convolution module
    network[f"{name}_conv_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }

    ## d --> 2d for GLU activation
    ## can linear as an alternative to pointwise conv.?
    network[f"{name}_conv_pointwise1"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_laynorm"],
        "n_out": 2 * model_dim,
        "forward_weights_init": initialization,
    }
    ## (batch, time, feature)
    network[f"{name}_conv_GLU"] = {
        "class": "gating",
        "activation": "identity",
        "from": [f"{name}_conv_pointwise1"],
    }

    if normal_conv:
        network[f"{name}_conv_expanded"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (-1, 1),
            "from": [f"{name}_conv_GLU"],
        }
        ## (T, F, 1)
        network[f"{name}_conv_normal"] = {
            "class": "conv",
            "from": [f"{name}_conv_expanded"],
            "padding": "same",
            "filter_size": (kernel_size, kernel_size_for_feature),
            "n_out": output_channels,
            "activation": None,
            "with_bias": True,  # model_dim//kernel_size
        }
        network[f"{name}_conv_normal_flattened"] = {
            "class": "merge_dims",
            "from": [f"{name}_conv_normal"],
            "axes": "static",
        }
        ## parameter intensiv
        network[f"{name}_conv_transformed"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "forward_weights_init": initialization,
            "n_out": model_dim,
            "from": [f"{name}_conv_normal_flattened"],
        }

        network[f"{name}_conv_{norm_in_between}"] = {
            "class": norm_in_between,
            "from": [f"{name}_conv_transformed"],
        }
    else:

        network[f"{name}_conv_depthwise"] = {
            "activation": None,
            "class": "conv",
            "filter_size": (kernel_size,),
            "from": [f"{name}_conv_GLU"],
            "groups": model_dim,
            "n_out": model_dim,
            "padding": "same",
            "with_bias": True,
        }

        network[f"{name}_conv_{norm_in_between}"] = {
            "class": norm_in_between,
            "from": [f"{name}_conv_depthwise"],
        }

    network[f"{name}_conv_act"] = {
        "class": "activation",
        "activation": "swish",
        "from": [f"{name}_conv_{norm_in_between}"],
    }

    network[f"{name}_conv_pointwise2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_act"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_conv_dropout"] = {
        "class": "dropout",
        "dropout": conv_post_dropout,
        "from": [f"{name}_conv_pointwise2"],
    }
    network[f"{name}_conv_output"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_conv_dropout"],
        "n_out": model_dim,
    }

    ## second ffn layer
    network[f"{name}_ff2_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_conv_output"],
    }
    network[f"{name}_ff2_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff2_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_ff2_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff2_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff2_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff2_conv2"],
    }

    network[f"{name}_ff2_drop_half"] = {
        "class": "eval",
        "eval": f"0.5 * source(0)",
        "from": [f"{name}_ff2_drop"],
    }
    network[f"{name}_ff2_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_conv_output", f"{name}_ff2_drop_half"],
    }

    if switch:
        network[f"{name}_conv_output"]["from"] = [
            f"{name}_ff1_out",
            f"{name}_conv_dropout",
        ]
        network[f"{name}_conv_laynorm"]["from"] = [f"{name}_ff1_out"]

        network[f"{name}_self_att_laynorm"]["from"] = [f"{name}_conv_output"]
        network[f"{name}_self_att_out"]["from"] = [
            f"{name}_conv_output",
            f"{name}_self_att_drop",
        ]

        network[f"{name}_ff2_laynorm"]["from"] = [f"{name}_self_att_out"]
        network[f"{name}_ff2_out"]["from"] = [
            f"{name}_self_att_out",
            f"{name}_ff2_drop_half",
        ]

    ## final layer norm
    if end_layernorm:
        network[f"{name}"] = {"class": "layer_norm", "from": [f"{name}_ff2_out"]}
    else:
        network[f"{name}"] = {"class": "copy", "from": [f"{name}_ff2_out"]}


def add_conformer_block(
    network,
    name,
    from_layer,
    ff_dim,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    kernel_size,
    output_channels,
    conv_post_dropout,
    initialization=DEFAULT_INIT,
    ff_activation="swish",
    normal_conv=False,
):
    ## first ffn with residual connection
    network[f"{name}_ff1_laynorm"] = {"class": "layer_norm", "from": [from_layer]}
    network[f"{name}_ff1_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff1_laynorm"],
        "dropout": ff_activation_dropout,
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_ff1_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff1_conv1"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff1_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff1_conv2"],
    }

    network[f"{name}_ff1_out"] = {
        "class": "eval",
        "eval": f"source(0) + 0.5 * source(1)",
        "from": [from_layer, f"{name}_ff1_drop"],
    }
    ## MHSA module
    network[f"{name}_self_att_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_ff1_out"],
    }
    network[f"{name}_self_att_att"] = {
        "class": "self_attention",
        "num_heads": num_heads,
        "total_key_dim": key_dim,
        "n_out": value_dim,
        "from": [f"{name}_self_att_laynorm"],
        "attention_dropout": sa_dropout,
        "forward_weights_init": initialization,
    }
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }

    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_ff1_out", f"{name}_self_att_drop"],
        "n_out": model_dim,
    }
    ## convolution module
    network[f"{name}_conv_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }

    ## d --> 2d for GLU activation
    ## can linear as an alternative to pointwise conv.?
    network[f"{name}_conv_pointwise1"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_laynorm"],
        "n_out": 2 * model_dim,
        "forward_weights_init": initialization,
    }
    ## (batch, time, feature)
    network[f"{name}_conv_GLU"] = {
        "class": "gating",
        "activation": "identity",
        "from": [f"{name}_conv_pointwise1"],
    }

    if normal_conv:
        ## (T, F) -> (T, F, 1)
        network[f"{name}_conv_expanded"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (-1, 1),
            "from": [f"{name}_conv_GLU"],
        }
        ## (T, F, 1) -> (T, F, output_channels)
        network[f"{name}_conv_normal"] = {
            "class": "conv",
            "from": [f"{name}_conv_expanded"],
            "padding": "same",
            "filter_size": (kernel_size, 3),
            "n_out": output_channels,
            "activation": None,
            "with_bias": True,  # model_dim//kernel_size
        }
        network[f"{name}_conv_normal_flattened"] = {
            "class": "merge_dims",
            "from": [f"{name}_conv_normal"],
            "axes": "static",
        }

        network[f"{name}_conv_transformed"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "forward_weights_init": initialization,
            "n_out": model_dim,
            "from": [f"{name}_conv_normal_flattened"],
        }

        network[f"{name}_conv_batchnorm"] = {
            "class": "batch_norm",
            "from": [f"{name}_conv_transformed"],
        }
    else:
        network[f"{name}_conv_depthwise"] = {
            "activation": None,
            "class": "conv",
            "filter_size": (kernel_size,),
            "from": [f"{name}_conv_GLU"],
            "groups": model_dim,
            "n_out": model_dim,
            "padding": "same",
            "with_bias": True,
        }

        network[f"{name}_conv_batchnorm"] = {
            "class": "batch_norm",
            "from": [f"{name}_conv_depthwise"],
        }

    network[f"{name}_conv_act"] = {
        "class": "activation",
        "activation": "swish",
        "from": [f"{name}_conv_batchnorm"],
    }

    network[f"{name}_conv_pointwise2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_act"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_conv_dropout"] = {
        "class": "dropout",
        "dropout": conv_post_dropout,
        "from": [f"{name}_conv_pointwise2"],
    }
    network[f"{name}_conv_output"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_conv_dropout"],
        "n_out": model_dim,
    }

    ## second ffn layer
    network[f"{name}_ff2_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_conv_output"],
    }
    network[f"{name}_ff2_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff2_laynorm"],
        "dropout": ff_activation_dropout,
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_ff2_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff2_conv1"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff2_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff2_conv2"],
    }

    network[f"{name}_ff2_out"] = {
        "class": "eval",
        "eval": f"source(0) + 0.5 * source(1)",
        "from": [f"{name}_conv_output", f"{name}_ff2_drop"],
    }
    ## final layer norm
    network[f"{name}"] = {"class": "layer_norm", "from": [f"{name}_ff2_out"]}


def add_conformer_block_with_l2(
    network,
    name,
    from_layer,
    ff_dim,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    kernel_size,
    output_channels,
    conv_post_dropout,
    initialization=DEFAULT_INIT,
    ff_activation="swish",
    normal_conv=False,
    l2=None,
):
    ## first ffn with residual connection
    network[f"{name}_ff1_laynorm"] = {"class": "layer_norm", "from": [from_layer]}
    network[f"{name}_ff1_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff1_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }

    network[f"{name}_ff1_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff1_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }
    network[f"{name}_ff1_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff1_conv2"],
    }

    network[f"{name}_ff1_out"] = {
        "class": "eval",
        "eval": f"source(0) + 0.5 * source(1)",
        "from": [from_layer, f"{name}_ff1_drop"],
    }
    ## MHSA module
    network[f"{name}_self_att_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_ff1_out"],
    }
    network[f"{name}_self_att_att"] = {
        "class": "self_attention",
        "num_heads": num_heads,
        "total_key_dim": key_dim,
        "n_out": value_dim,
        "from": [f"{name}_self_att_laynorm"],
        "attention_dropout": sa_dropout,
        "forward_weights_init": initialization,
        "L2": l2,
    }
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }

    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_ff1_out", f"{name}_self_att_drop"],
        "n_out": model_dim,
    }
    ## convolution module
    network[f"{name}_conv_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }

    ## d --> 2d for GLU activation
    ## can linear as an alternative to pointwise conv.?
    network[f"{name}_conv_pointwise1"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_laynorm"],
        "n_out": 2 * model_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }
    ## (batch, time, feature)
    network[f"{name}_conv_GLU"] = {
        "class": "gating",
        "activation": "identity",
        "from": [f"{name}_conv_pointwise1"],
    }

    if normal_conv:
        network[f"{name}_conv_expanded"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (-1, 1),
            "from": [f"{name}_conv_GLU"],
        }

        network[f"{name}_conv_normal"] = {
            "class": "conv",
            "from": [f"{name}_conv_expanded"],
            "padding": "same",
            "filter_size": (kernel_size, 3),
            "n_out": output_channels,
            "activation": None,
            "with_bias": True,  # model_dim//kernel_size
            "L2": l2,
        }
        network[f"{name}_conv_normal_flattened"] = {
            "class": "merge_dims",
            "from": [f"{name}_conv_normal"],
            "axes": "static",
        }

        network[f"{name}_conv_transformed"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "forward_weights_init": initialization,
            "n_out": model_dim,
            "from": [f"{name}_conv_normal_flattened"],
            "L2": l2,
        }

        network[f"{name}_conv_batchnorm"] = {
            "class": "batch_norm",
            "from": [f"{name}_conv_transformed"],
        }
    else:

        network[f"{name}_conv_depthwise"] = {
            "activation": None,
            "class": "conv",
            "filter_size": (kernel_size,),
            "from": [f"{name}_conv_GLU"],
            "groups": model_dim,
            "n_out": model_dim,
            "padding": "same",
            "with_bias": True,
            "L2": l2,
        }

        network[f"{name}_conv_batchnorm"] = {
            "class": "batch_norm",
            "from": [f"{name}_conv_depthwise"],
        }

    network[f"{name}_conv_act"] = {
        "class": "activation",
        "activation": "swish",
        "from": [f"{name}_conv_batchnorm"],
    }

    network[f"{name}_conv_pointwise2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_act"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }

    network[f"{name}_conv_dropout"] = {
        "class": "dropout",
        "dropout": conv_post_dropout,
        "from": [f"{name}_conv_pointwise2"],
    }
    network[f"{name}_conv_output"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_conv_dropout"],
        "n_out": model_dim,
    }

    ## second ffn layer
    network[f"{name}_ff2_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_conv_output"],
    }
    network[f"{name}_ff2_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff2_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }

    network[f"{name}_ff2_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff2_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
        "L2": l2,
    }
    network[f"{name}_ff2_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff2_conv2"],
    }

    network[f"{name}_ff2_out"] = {
        "class": "eval",
        "eval": f"source(0) + 0.5 * source(1)",
        "from": [f"{name}_conv_output", f"{name}_ff2_drop"],
    }
    ## final layer norm
    # network[f"{name}"] = {
    #   'class': "layer_norm",
    #   'from': [f"{name}_ff2_out"]
    # }
    network[f"{name}"] = {"class": "copy", "from": [f"{name}_ff2_out"]}


def add_conformer_block_with_relative_pos(
    network,
    name,
    from_layer,
    ff_dim,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    kernel_size,
    output_channels,
    conv_post_dropout,
    initialization=DEFAULT_INIT,
    ff_activation="swish",
    normal_conv=False,
    clipping=100,
    fixed=False,
):
    ## first ffn with residual connection
    network[f"{name}_ff1_laynorm"] = {"class": "layer_norm", "from": [from_layer]}
    network[f"{name}_ff1_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff1_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_ff1_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff1_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff1_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff1_conv2"],
    }

    network[f"{name}_ff1_out"] = {
        "class": "eval",
        "eval": f"source(0) + 0.5 * source(1)",
        "from": [from_layer, f"{name}_ff1_drop"],
    }
    ## MHSA module
    network[f"{name}_self_att_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_ff1_out"],
    }

    network[f"{name}_rel_pos"] = {
        "class": "relative_positional_encoding",
        "from": [f"{name}_self_att_laynorm"],
        "fixed": fixed,
        "clipping": clipping,
        "n_out": key_dim // num_heads,
        "forward_weights_init": initialization,
    }

    network[f"{name}_self_att_att"] = {
        "class": "self_attention",
        "num_heads": num_heads,
        "total_key_dim": key_dim,
        "n_out": value_dim,
        "from": [f"{name}_self_att_laynorm"],
        "attention_dropout": sa_dropout,
        "forward_weights_init": initialization,
        "key_shift": f"{name}_rel_pos",
    }
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }

    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_ff1_out", f"{name}_self_att_drop"],
        "n_out": model_dim,
    }
    ## convolution module
    network[f"{name}_conv_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }

    ## d --> 2d for GLU activation
    ## can linear as an alternative to pointwise conv.?
    network[f"{name}_conv_pointwise1"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_laynorm"],
        "n_out": 2 * model_dim,
        "forward_weights_init": initialization,
    }
    ## (batch, time, feature)
    network[f"{name}_conv_GLU"] = {
        "class": "gating",
        "activation": "identity",
        "from": [f"{name}_conv_pointwise1"],
    }

    if normal_conv:
        network[f"{name}_conv_expanded"] = {
            "class": "split_dims",
            "axis": "F",
            "dims": (-1, 1),
            "from": [f"{name}_conv_GLU"],
        }

        network[f"{name}_conv_normal"] = {
            "class": "conv",
            "from": [f"{name}_conv_expanded"],
            "padding": "same",
            "filter_size": (kernel_size, 3),
            "n_out": output_channels,
            "activation": None,
            "with_bias": True,  # model_dim//kernel_size
        }
        network[f"{name}_conv_normal_flattened"] = {
            "class": "merge_dims",
            "from": [f"{name}_conv_normal"],
            "axes": "static",
        }

        network[f"{name}_conv_transformed"] = {
            "class": "linear",
            "activation": None,
            "with_bias": False,
            "forward_weights_init": initialization,
            "n_out": model_dim,
            "from": [f"{name}_conv_normal_flattened"],
        }

        network[f"{name}_conv_batchnorm"] = {
            "class": "batch_norm",
            "from": [f"{name}_conv_transformed"],
        }
    else:
        network[f"{name}_conv_depthwise"] = {
            "activation": None,
            "class": "conv",
            "filter_size": (kernel_size,),
            "from": [f"{name}_conv_GLU"],
            "groups": model_dim,
            "n_out": model_dim,
            "padding": "same",
            "with_bias": True,
        }
        network[f"{name}_conv_batchnorm"] = {
            "class": "batch_norm",
            "from": [f"{name}_conv_depthwise"],
        }

    network[f"{name}_conv_act"] = {
        "class": "activation",
        "activation": "swish",
        "from": [f"{name}_conv_batchnorm"],
    }

    network[f"{name}_conv_pointwise2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_conv_act"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_conv_dropout"] = {
        "class": "dropout",
        "dropout": conv_post_dropout,
        "from": [f"{name}_conv_pointwise2"],
    }
    network[f"{name}_conv_output"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_conv_dropout"],
        "n_out": model_dim,
    }

    ## second ffn layer
    network[f"{name}_ff2_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_conv_output"],
    }
    network[f"{name}_ff2_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff2_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }

    network[f"{name}_ff2_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff2_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff2_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff2_conv2"],
    }

    network[f"{name}_ff2_out"] = {
        "class": "eval",
        "eval": f"source(0) + 0.5 * source(1)",
        "from": [f"{name}_conv_output", f"{name}_ff2_drop"],
    }
    ## final layer norm
    network[f"{name}"] = {"class": "layer_norm", "from": [f"{name}_ff2_out"]}


## should implement masked multi-head self-attention, encoder decoder attention, FFN sub-layer
## 'name' is used to name all the layers
## does 'num_heads' determine both the self-attention and encoder-decoder attention?
def trafo_dec_layer(
    network,
    subnetwork,
    name,
    num_heads,
    key_dim,
    value_dim,
    model_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    att_dropout,
    att_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
):
    key_per_head = key_dim // num_heads
    value_per_head = value_dim // num_heads

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # decoder self attention block
    ## again start by a layer normalization
    subnetwork[f"{name}_self_att_laynorm"] = {
        "class": "layer_norm",
        "from": from_layers,
    }
    ## self-attention layer
    subnetwork[f"{name}_self_att_att"] = {
        "class": "self_attention",
        "num_heads": num_heads,
        "total_key_dim": key_dim,
        "n_out": value_dim,
        "from": [f"{name}_self_att_laynorm"],
        "attention_left_only": True,
        "attention_dropout": sa_dropout,
        "forward_weights_init": initialization,
    }
    ## linear transformation after self-attention
    subnetwork[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    ## drop out
    subnetwork[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    subnetwork[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,
    }

    # decoder attention block

    ## layer normalization
    subnetwork[f"{name}_att_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }
    ## transform the decoder side self-attention to query
    subnetwork[f"{name}_att_query0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_att_laynorm"],
        "n_out": value_dim,
        "forward_weights_init": initialization,
    }
    subnetwork[f"{name}_att_query"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),  # (B, H, D/H)
        "from": [f"{name}_att_query0"],
    }

    ## the subnetwork is attached to the base network where the keys and values from the encoder outputs are computed
    subnetwork[f"{name}_att_energy"] = {
        "class": "dot",
        "red1": -1,
        "red2": -1,
        "var1": "T",
        "var2": "T?",
        "from": [f"base:{name}_att_key", f"{name}_att_query"],
    }  # (B, H, enc-T, 1)
    ## for each head, each encoder output there is an energy score
    ## the next step is to compute the probs. along the enc-T dimension
    subnetwork[f"{name}_att_weights"] = {
        "class": "softmax_over_spatial",
        "from": [f"{name}_att_energy"],
        "energy_factor": key_per_head**-0.5,  # (B, enc-T, H, 1)
    }
    ## drop out for encoder-decoder attention
    subnetwork[f"{name}_att_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "dropout": att_dropout,
        "from": [f"{name}_att_weights"],
    }
    ## get the context vector for current step
    subnetwork[f"{name}_att0"] = {
        "class": "generic_attention",
        "weights": f"{name}_att_weights_drop",
        "base": f"base:{name}_att_value",  # (B, H, V)
    }
    subnetwork[f"{name}_att_att"] = {
        "class": "merge_dims",
        "axes": "static",
        "from": [f"{name}_att0"],
    }  # (B, H*V)

    ## map the concatenated multi-head encoder decoder outputs to a fixed dim. value_dim
    subnetwork[f"{name}_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_att_att"],
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    subnetwork[f"{name}_att_drop"] = {
        "class": "dropout",
        "dropout": att_post_dropout,
        "from": [f"{name}_att_lin"],
    }
    ## residual connection from the decoder self attention outputs
    subnetwork[f"{name}_att_out"] = {
        "class": "combine",
        "kind": "add",
        "n_out": model_dim,
        "from": [f"{name}_att_drop", f"{name}_self_att_out"],
    }

    # network attention block
    ## the encoder outputs are transformed to keys: key_dim
    ## so the last layer of the encoder must be named with 'encoder'?
    ## the name of the layers corresponds to the decoder block because for each decoder block
    ## a key and value mapping is applied
    network[f"{name}_att_key0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": ["encoder"],
        "n_out": key_dim,  # (B, enc-T, D)
        "forward_weights_init": initialization,
    }
    ## transform the encoder outputs to values: value_dim
    network[f"{name}_att_value0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": ["encoder"],
        "n_out": value_dim,
        "forward_weights_init": initialization,
    }

    ## kind weired, so why are the two kinds of multi-head attention outputs in the decoder not split beforehand
    network[f"{name}_att_key"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),
        "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
    }
    network[f"{name}_att_value"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, value_per_head),
        "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
    }

    # decoder feed forward block
    subnetwork[f"{name}_ff_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_att_out"],
    }
    subnetwork[f"{name}_ff_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }
    subnetwork[f"{name}_ff_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    subnetwork[f"{name}_ff_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff_conv2"],
    }
    subnetwork[f"{name}_ff_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_att_out", f"{name}_ff_drop"],
        "n_out": model_dim,
    }
    subnetwork[f"{name}"] = {"class": "copy", "from": [f"{name}_ff_out"]}

    return subnetwork


def separated_trafo_ca_layer(
    network,
    name,
    num_heads,
    model_dim,
    key_dim,
    value_dim,
    ff_dim,
    sa_dropout,
    sa_post_dropout,
    ff_activation_dropout,
    ff_post_dropout,
    from_layers,
    ca_layer,
    initialization=DEFAULT_INIT,
    ff_activation="relu",
):
    key_per_head = int(key_dim / num_heads)
    value_per_head = int(value_dim / num_heads)

    if from_layers is None:
        from_layers = ["data"]
    elif isinstance(from_layers, str):
        from_layers = [from_layers]

    # self attention block
    ## first add a layer normalization
    network[f"{name}_self_att_laynorm"] = {"class": "layer_norm", "from": from_layers}

    network[f"{name}_att_query0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,
        "forward_weights_init": initialization,
    }

    # query per head
    network[f"{name}_att_query"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
        "from": [f"{name}_att_query0"],
    }

    network[f"{name}_att_key0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": key_dim,  # (B, enc-T, D)
        "forward_weights_init": initialization,
    }
    network[f"{name}_att_value0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_laynorm"],
        "n_out": value_dim,
        "forward_weights_init": initialization,
    }

    ## split the key and value vectors for each head
    network[f"{name}_att_key"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),
        "from": [f"{name}_att_key0"],  # (B, enc-T, H, D/H)
    }

    network[f"{name}_att_value"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, value_per_head),
        "from": [f"{name}_att_value0"],  # (B, enc-T, H, D'/H)
    }

    ## encoder-decoder energy
    ## we have exactly enc-T energy values
    network[f"{name}_att_energy"] = {
        "class": "dot",
        "red1": -1,
        "red2": -1,
        "var1": "T",
        "var2": "T?",
        "from": [f"{name}_att_key", f"{name}_att_query"],
    }  # (B, H, enc-T, enc-T) #(B, H, enc-T, 1)

    ## normalize the attention weights (depends on key/query dim.)
    network[f"{name}_att_weights"] = {
        "class": "softmax_over_spatial",
        "from": [f"{name}_att_energy"],
        "energy_factor": key_per_head**-0.5,  # (B, enc-T, H, 1)
    }

    ## attention weights dropout
    network[f"{name}_att_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "dropout": sa_dropout,
        "from": [f"{name}_att_weights"],
    }

    ## now we have an attention weight value for each encoder-side output
    ## we get per head one vector
    network[f"{name}_att0"] = {
        "class": "generic_attention",
        "weights": f"{name}_att_weights_drop",
        "base": f"{name}_att_value",  # (B, T, H, V) #(B, H, V)
    }

    network[f"{name}_self_att_att"] = {
        "class": "merge_dims",
        "axes": "static",  # "static"
        "from": [f"{name}_att0"],
    }

    ## one multi-head attention layer
    ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)

    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_self_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_self_att_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_self_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_self_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_self_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": from_layers + [f"{name}_self_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    #####################################################

    ## cross-attention block
    network[f"{name}_cross_att_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_self_att_out"],
    }
    ## only for generating queries
    network[f"{name}_cross_att_query0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_cross_att_laynorm"],
        "n_out": key_dim,
        "forward_weights_init": initialization,
    }

    # query per head
    network[f"{name}_cross_att_query"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),  # (B, T, H, D/H)
        "from": [f"{name}_cross_att_query0"],
    }

    ## key from ca_layer
    network[f"{name}_cross_att_key0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [ca_layer],
        "n_out": key_dim,  # (B, enc-T, D)
        "forward_weights_init": initialization,
    }
    ## value also from ca layer
    network[f"{name}_cross_att_value0"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [ca_layer],
        "n_out": value_dim,
        "forward_weights_init": initialization,
    }

    ## split the key and value vectors for each head
    network[f"{name}_cross_att_key"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, key_per_head),
        "from": [f"{name}_cross_att_key0"],  # (B, enc-T, H, D/H)
    }

    network[f"{name}_cross_att_value"] = {
        "class": "split_dims",
        "axis": "F",
        "dims": (num_heads, value_per_head),
        "from": [f"{name}_cross_att_value0"],  # (B, enc-T, H, D'/H)
    }

    ## cross energy
    ## we have exactly enc-T energy values

    network[f"{name}_cross_att_energy"] = {
        "class": "dot",
        "red1": -1,
        "red2": -1,
        "var1": "T",
        "var2": "T?",
        "from": [f"{name}_cross_att_key", f"{name}_cross_att_query"],
    }  # (B, H, enc-T/2, enc-T) <-- (B, H, enc-T, enc-T) #(B, H, enc-T, 1)

    ## normalize the attention weights (depends on key/query dim.)
    network[f"{name}_cross_att_weights"] = {
        "class": "softmax_over_spatial",
        "from": [f"{name}_cross_att_energy"],
        "energy_factor": key_per_head**-0.5,  # (B, enc-T, H, 1)
    }

    ## attention weights dropout
    network[f"{name}_cross_att_weights_drop"] = {
        "class": "dropout",
        "dropout_noise_shape": {"*": None},
        "dropout": sa_dropout,
        "from": [f"{name}_cross_att_weights"],
    }

    ## now we have an attention weight value for each encoder-side output
    ## we get per head one vector
    network[f"{name}_cross_att0"] = {
        "class": "generic_attention",
        "weights": f"{name}_cross_att_weights_drop",
        "base": f"{name}_cross_att_value0",  # (B, T, H, V) # (B, H, V)
    }

    network[f"{name}_cross_att"] = {
        "class": "merge_dims",
        "axes": "static",  # "static"
        "from": [f"{name}_cross_att0"],
    }

    ## one multi-head attention layer
    ## 1: key dim. (sum of total num_heads key dims.), 2: value dim. (sum over heads)

    ## a linear transformation layer (value_dim to model_dim)
    network[f"{name}_cross_att_lin"] = {
        "class": "linear",
        "activation": None,
        "with_bias": False,
        "from": [f"{name}_cross_att"],
        "n_out": model_dim,  # value_dim,
        "forward_weights_init": initialization,
    }
    ## dropout after linear transformation of the multi-head outputs
    network[f"{name}_cross_att_drop"] = {
        "class": "dropout",
        "dropout": sa_post_dropout,
        "from": [f"{name}_cross_att_lin"],
    }
    ## residual connection
    ## so the input to the transformer block should also be model_dim dim.
    network[f"{name}_cross_att_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_self_att_out", f"{name}_cross_att_drop"],
        "n_out": model_dim,  # value_dim,
    }

    # feed forward block
    ## two linear layers with activation in between
    ## ff_dim would be the size of hidden units
    ## the output of the FNN sub-layer would be input to the next transformer block and
    ## therefore the output dim. should be model_dim
    network[f"{name}_ff_laynorm"] = {
        "class": "layer_norm",
        "from": [f"{name}_cross_att_out"],
    }
    network[f"{name}_ff_conv1"] = {
        "class": "linear",
        "activation": ff_activation,
        "with_bias": True,
        "from": [f"{name}_ff_laynorm"],
        "n_out": ff_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_conv2"] = {
        "class": "linear",
        "activation": None,
        "with_bias": True,
        "from": [f"{name}_ff_conv1"],
        "dropout": ff_activation_dropout,
        "n_out": model_dim,
        "forward_weights_init": initialization,
    }
    network[f"{name}_ff_drop"] = {
        "class": "dropout",
        "dropout": ff_post_dropout,
        "from": [f"{name}_ff_conv2"],
    }
    network[f"{name}_ff_out"] = {
        "class": "combine",
        "kind": "add",
        "from": [f"{name}_cross_att_out", f"{name}_ff_drop"],
        "n_out": model_dim,  # value_dim,
    }
    network[f"{name}"] = {"class": "layer_norm", "from": [f"{name}_ff_out"]}

    return network
