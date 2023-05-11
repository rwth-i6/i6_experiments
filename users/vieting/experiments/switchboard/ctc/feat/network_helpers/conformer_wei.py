"""
Helper to create network dictionaries.

Taken from Wei's setup without too much cleanup.
"""
import copy


def add_loss_to_layer(network, name, loss, loss_opts=None, target=None, **kwargs):
    assert loss is not None
    network[name]["loss"] = loss
    if loss_opts:
        network[name]["loss_opts"] = loss_opts
    if target is not None:
        network[name]["target"] = target
    return network


def add_specaug_source_layer(network, name="source", next_layers=None, source=None):
    next_layers = next_layers or ["fwd_lstm_1", "bwd_lstm_1"]
    assert next_layers is not None
    network2 = copy.deepcopy(network)
    network2[name] = {
        "class": "eval",
        "eval": "self.network.get_config().typed_value('transform')(source(0, as_data=True), network=self.network)",
    }
    if source is not None:
        network2[name]["from"] = source
    for layer in next_layers:
        if layer not in network2:
            continue
        network2[layer]["from"] = name
    return network2, name


def add_linear_layer(network, name, from_list, size, l2=0.01, dropout=None, bias=None, activation=None, **kwargs):
    network[name] = {"class": "linear", "n_out": size, "from": from_list, "activation": activation}
    if l2 is not None:
        network[name]["L2"] = l2
    if dropout is not None:
        network[name]["dropout"] = dropout
    # bias is default true in RETURNN
    if bias is not None:
        network[name]["with_bias"] = bias
    if kwargs.get("random_norm_init", False):
        network[name]["forward_weights_init"] = "random_normal_initializer(mean=0.0, stddev=0.1)"
    if kwargs.get("initial", None) is not None:
        network[name]["initial_output"] = kwargs.get("initial", None)
    if kwargs.get("loss", None) is not None:
        network = add_loss_to_layer(network, name, **kwargs)
    if kwargs.get("reuse_params", None) is not None:
        network[name]["reuse_params"] = kwargs.get("reuse_params", None)
    if not kwargs.get("trainable", True):
        network[name]["trainable"] = False
    if kwargs.get("out_type", None) is not None:
        network[name]["out_type"] = kwargs.get("out_type", None)
    # Note: this is not in the master RETURNN branch
    if kwargs.get("safe_embedding", False):
        network[name]["safe_embedding"] = True  # 0-vectors for out-of-range ids (only for embedding)
    if kwargs.get("validate_indices", False):
        network[name]["validate_indices"] = True  # round out-of-range ids to 0 (only for embedding)
    return network, name


def add_activation_layer(network, name, from_list, activation, **kwargs):
    network[name] = {"class": "activation", "from": from_list, "activation": activation}
    if kwargs.get("loss", None) is not None:
        network = add_loss_to_layer(network, name, **kwargs)
    return network, name


def add_copy_layer(network, name, from_list, initial=None, loss=None, **kwargs):
    network[name] = {"class": "copy", "from": from_list}
    if initial is not None:
        network[name]["initial_output"] = initial
    if loss is not None:
        network = add_loss_to_layer(network, name, loss, **kwargs)
    if kwargs.get("is_output", False):
        network[name]["is_output_layer"] = True
    if kwargs.get("dropout", None) is not None:
        network[name]["dropout"] = kwargs.get("dropout", None)
    return network, name


def add_pool_layer(network, name, from_list, mode="max", pool_size=(2,), padding="same", **kwargs):
    network[name] = {
        "class": "pool",
        "mode": mode,
        "padding": padding,
        "pool_size": pool_size,
        "from": from_list,
        "trainable": False,
    }
    return network, name


def add_merge_dim_layer(network, name, from_list, axes="except_time", **kwargs):
    network[name] = {"class": "merge_dims", "from": from_list, "axes": axes}
    return network, name


def add_split_dim_layer(network, name, from_list, axis, dims, **kwargs):
    network[name] = {"class": "split_dims", "from": from_list, "axis": axis, "dims": dims}
    return network, name


def add_layer_norm_layer(network, name, from_list, **kwargs):
    network[name] = {"class": "layer_norm", "from": from_list}
    if not kwargs.get("trainable", True):
        network[name]["trainable"] = False
    return network, name


def add_batch_norm_layer(network, name, from_list, **kwargs):
    network[name] = {"class": "batch_norm", "from": from_list}
    # RETURNN defaults wrong
    if kwargs.get("fix_settings", False):
        network[name].update(
            {
                "momentum": 0.1,
                "epsilon": 1e-5,
                # otherwise eval may be batch-size and utterance-order dependent !
                "update_sample_only_in_training": True,
                "delay_sample_update": True,
            }
        )
    # freeze batch norm running average in training: consistent with testing
    if kwargs.get("freeze_average", False):
        network[name]["momentum"] = 0.0
        network[name]["use_sample"] = 1.0
    if not kwargs.get("trainable", True):
        network[name]["trainable"] = False
    return network, name


# eval layer is a also special case of combine layer, but we distinguish them explicitly here
# and only restricted to the 'kind' usage
def add_combine_layer(network, name, from_list, kind="add", **kwargs):
    network[name] = {"class": "combine", "from": from_list, "kind": kind}
    if kwargs.get("activation", None) is not None:
        network[name]["activation"] = kwargs.get("activation", None)
    if kwargs.get("with_bias", None) is not None:
        network[name]["with_bias"] = kwargs.get("with_bias", None)
    if kwargs.get("n_out", None) is not None:
        network[name]["n_out"] = kwargs.get("n_out", None)
    if kwargs.get("out_type", None) is not None:
        network[name]["out_type"] = kwargs.get("out_type", None)
    if kwargs.get("is_output", False):
        network[name]["is_output_layer"] = True
    return network, name


# Note: RETURNN source(i, auto_convert=True, enforce_batch_major=False, as_data=False)
def add_eval_layer(network, name, from_list, eval_str, **kwargs):
    network[name] = {"class": "eval", "from": from_list, "eval": eval_str}
    if kwargs.get("loss", None) is not None:
        network = add_loss_to_layer(network, name, **kwargs)
    if kwargs.get("initial", None) is not None:
        network[name]["initial_output"] = kwargs.get("initial", None)
    if kwargs.get("n_out", None) is not None:
        network[name]["n_out"] = kwargs.get("n_out", None)
    if kwargs.get("out_type", None) is not None:
        network[name]["out_type"] = kwargs.get("out_type", None)
    if kwargs.get("is_output", False):
        network[name]["is_output_layer"] = True
    return network, name


def add_rel_pos_encoding_layer(network, name, from_list, n_out, clipping=64, **kwargs):
    network[name] = {"class": "relative_positional_encoding", "from": from_list, "n_out": n_out, "clipping": clipping}
    if not kwargs.get("trainable", True):
        network[name]["trainable"] = False
    return network, name


def add_self_attention_layer(
    network, name, from_list, n_out, num_heads, total_key_dim, key_shift=None, attention_dropout=None, **kwargs
):
    network[name] = {
        "class": "self_attention",
        "from": from_list,
        "n_out": n_out,
        "num_heads": num_heads,
        "total_key_dim": total_key_dim,
    }
    if key_shift is not None:
        network[name]["key_shift"] = key_shift
    if attention_dropout is not None:
        network[name]["attention_dropout"] = attention_dropout
    if not kwargs.get("trainable", True):
        network[name]["trainable"] = False
    return network, name


def add_conv_layer(
    network, name, from_list, n_out, filter_size, padding="VALID", l2=0.01, bias=True, activation=None, **kwargs
):
    network[name] = {
        "class": "conv",
        "from": from_list,
        "n_out": n_out,
        "filter_size": filter_size,
        "padding": padding,
        "with_bias": bias,
        "activation": activation,
    }
    if l2 is not None:
        network[name]["L2"] = l2
    if kwargs.get("strides", None) is not None:
        network[name]["strides"] = kwargs.get("strides", None)
    if kwargs.get("groups", None) is not None:
        network[name]["groups"] = kwargs.get("groups", None)
    if not kwargs.get("trainable", True):
        network[name]["trainable"] = False
    return network, name


def add_gating_layer(network, name, from_list, activation=None, gate_activation="sigmoid", **kwargs):
    network[name] = {"class": "gating", "from": from_list, "activation": activation, "gate_activation": gate_activation}
    return network, name


# Convolution block
def add_conv_block(
    network, from_list, conv_layers, conv_filter, conv_size, pool_size=None, name_prefix="conv", **kwargs
):
    network, from_list = add_split_dim_layer(network, "conv_source", from_list, axis="F", dims=(-1, 1))
    for idx in range(conv_layers):
        name = name_prefix + "_" + str(idx + 1)
        network, from_list = add_conv_layer(network, name, from_list, conv_size, conv_filter, padding="same", **kwargs)
        if pool_size is not None:
            name += "_pool"
            if isinstance(pool_size, list):
                assert idx < len(pool_size)
                pool = pool_size[idx]
            else:
                pool = pool_size
            assert isinstance(pool, tuple)
            if any([p > 1 for p in pool]):
                network, from_list = add_pool_layer(network, name, from_list, pool_size=pool)
    network, from_list = add_merge_dim_layer(network, "conv_merged", from_list, axes="static")
    return network, from_list


# Conformer encoder
def add_conformer_block(network, name, from_list, size, dropout, l2, trainable=True, **kwargs):
    # feed-forward module
    def add_ff_module(net, n, fin):
        net, fout = add_layer_norm_layer(net, n + "_ln", fin, trainable=trainable)
        net, fout = add_linear_layer(
            net, n + "_linear_swish", fout, size * 4, l2=l2, activation="swish", trainable=trainable
        )
        net, fout = add_linear_layer(
            net, n + "_dropout_linear", fout, size, l2=l2, dropout=dropout, trainable=trainable
        )
        net, fout = add_copy_layer(net, n + "_dropout", fout, dropout=dropout)
        net, fout = add_eval_layer(net, n + "_half_res_add", [fout, fin], "0.5 * source(0) + source(1)")
        return net, fout

    # multi-head self-attention module
    def add_mhsa_module(net, n, fin, heads, pos_enc_size, pos_enc_clip, pos_enc=True):
        net, fout = add_layer_norm_layer(net, n + "_ln", fin, trainable=trainable)
        if pos_enc:
            net, fpos = add_rel_pos_encoding_layer(
                net, n + "_relpos_encoding", fout, pos_enc_size, clipping=pos_enc_clip, trainable=trainable
            )
        else:
            fpos = None
        net, fout = add_self_attention_layer(
            net,
            n + "_self_attention",
            fout,
            size,
            heads,
            size,
            key_shift=fpos,
            attention_dropout=dropout,
            trainable=trainable,
        )
        net, fout = add_linear_layer(net, n + "_att_linear", fout, size, l2=l2, bias=False, trainable=trainable)
        net, fout = add_copy_layer(net, n + "_dropout", fout, dropout=dropout)
        net, fout = add_combine_layer(net, n + "_res_add", [fout, fin])
        return net, fout

    # convolution module
    def add_conv_module(net, n, fin, filter_size, bn_fix, bn_freeze, bn2ln):
        net, fout = add_layer_norm_layer(net, n + "_ln", fin, trainable=trainable)
        # glu weights merged into pointwise conv, i.e. linear layer
        net, fout = add_linear_layer(net, n + "_pointwise_conv_1", fout, size * 2, l2=l2, trainable=trainable)
        net, fout = add_gating_layer(net, n + "_glu", fout)
        net, fout = add_conv_layer(
            net, n + "_depthwise_conv", fout, size, filter_size, padding="same", l2=l2, groups=size, trainable=trainable
        )
        if bn2ln:
            net, fout = add_layer_norm_layer(net, n + "_bn2ln", fout, trainable=trainable)
        else:
            net, fout = add_batch_norm_layer(
                net, n + "_bn", fout, fix_settings=bn_fix, freeze_average=bn_freeze, trainable=trainable
            )
        net, fout = add_activation_layer(net, n + "_swish", fout, "swish")
        net, fout = add_linear_layer(net, n + "_pointwise_conv_2", fout, size, l2=l2, trainable=trainable)
        net, fout = add_copy_layer(net, n + "_dropout", fout, dropout=dropout)
        net, fout = add_combine_layer(net, n + "_res_add", [fout, fin])
        return net, fout

    network, f_list = add_ff_module(network, name + "_ffmod_1", from_list)

    mhsa_args = {
        "heads": kwargs.get("num_att_heads", 8),
        "pos_enc_size": kwargs.get("pos_enc_size", 64),
        "pos_enc_clip": kwargs.get("pos_enc_clip", 64),  # default clipping 16 in RETURNN
        "pos_enc": kwargs.get("pos_encoding", True),
    }
    conv_args = {
        "filter_size": kwargs.get("conv_filter_size", (32,)),
        "bn_fix": kwargs.get("batch_norm_fix", False),
        "bn_freeze": kwargs.get("batch_norm_freeze", False),
        "bn2ln": kwargs.get("batch_norm_to_layer_norm", False),
    }
    if kwargs.get("switch_conv_mhsa_module", False):
        network, f_list = add_conv_module(network, name + "_conv_mod", f_list, **conv_args)
        network, f_list = add_mhsa_module(network, name + "_mhsa_mod", f_list, **mhsa_args)
    else:
        network, f_list = add_mhsa_module(network, name + "_mhsa_mod", f_list, **mhsa_args)
        network, f_list = add_conv_module(network, name + "_conv_mod", f_list, **conv_args)

    network, f_list = add_ff_module(network, name + "_ffmod_2", f_list)
    network, f_list = add_layer_norm_layer(network, name + "_output", f_list, trainable=trainable)
    return network, f_list


def build_conformer_encoder(
    network, from_list, num_blocks=12, size=512, dropout=0.1, l2=0.0001, max_pool=None, **kwargs
):
    max_pool = max_pool or []
    network, from_list = add_linear_layer(
        network, "input_linear", from_list, size, l2=l2, bias=False, trainable=kwargs.get("trainable", True)
    )
    network, from_list = add_copy_layer(network, "input_dropout", from_list, dropout=dropout)

    # Conformer blocks
    for idx in range(num_blocks):
        name = "conformer_" + str(idx + 1)
        network, from_list = add_conformer_block(network, name, from_list, size, dropout, l2, **kwargs)
        # also allow subsampling between conformer blocks
        if max_pool and idx < len(max_pool) and max_pool[idx] > 1:
            name += "_max_pool"
            network, from_list = add_pool_layer(network, name, from_list, pool_size=(max_pool[idx],))
    return network, from_list


def add_conformer_stack(network, from_list, dropout=0.1, l2=1e-4, encoder_layers=12, encoder_size=512, **kwargs):
    network, from_list = build_conformer_encoder(
        network,
        from_list,
        num_blocks=encoder_layers,
        size=encoder_size,
        dropout=dropout,
        l2=l2,
        **kwargs,
    )

    return network, from_list


def add_vgg_stack(network, from_list, **kwargs):
    network, from_list = add_conv_block(
        network, from_list, 1, (3, 3), 32, pool_size=(1, 2), activation="swish", **kwargs
    )
    stride1, stride2 = kwargs.get("vgg_conv_strides", (2, 2))
    network, f_list = add_conv_layer(
        network,
        "conv_2",
        network[from_list]["from"],
        64,
        (3, 3),
        padding="same",
        strides=(stride1, 1),
        activation="swish",
        **kwargs,
    )
    network, f_list = add_conv_layer(
        network, "conv_3", f_list, 64, (3, 3), padding="same", strides=(stride2, 1), activation="swish", **kwargs
    )
    network[from_list]["from"] = f_list
    return network, from_list
