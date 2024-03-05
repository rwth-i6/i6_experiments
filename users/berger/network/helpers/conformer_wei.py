from typing import Any, Dict, List, Optional, Tuple, Union


def get_variance_scaling_init(scale: float = 0.78) -> Dict[str, Any]:
    return {
        "class": "VarianceScaling",
        "distribution": "uniform",
        "mode": "fan_in",
        "scale": scale,
    }


def add_ff_module(
    network: Dict,
    name: str,
    from_list: Union[str, List[str]],
    size: int = 512,
    dropout: float = 0.1,
    l2: float = 5e-06,
    reuse_from_name: Optional[str] = None,
    ff_half_res_add: bool = True,
    **kwargs,
) -> str:
    network.update(
        {
            f"{name}_input": {
                "class": "copy",
                "from": from_list,
            },
            f"{name}_ln": {
                "class": "layer_norm",
                "from": f"{name}_input",
            },
            f"{name}_ff_1": {
                "class": "linear",
                "from": f"{name}_ln",
                "n_out": size * 4,
                "activation": "swish",
                "forward_weights_init": get_variance_scaling_init(),
                "L2": l2,
            },
            f"{name}_ff_2": {
                "class": "linear",
                "activation": None,
                "from": f"{name}_ff_1",
                "n_out": size,
                "L2": l2,
                "forward_weights_init": get_variance_scaling_init(),
                "dropout": dropout,
            },
            f"{name}_dropout": {
                "class": "copy",
                "from": f"{name}_ff_2",
                "dropout": dropout,
            },
            f"{name}_res_add": {
                "from": [f"{name}_dropout", f"{name}_input"],
            },
        }
    )

    if ff_half_res_add:
        network[f"{name}_res_add"].update(
            {
                "class": "eval",
                "eval": "0.5 * source(0) + source(1)",
            }
        )
    else:
        network[f"{name}_res_add"].update(
            {
                "class": "combine",
                "kind": "add",
            }
        )

    if reuse_from_name:
        for suffix in ["_ff_1", "_ff_2"]:
            network[name + suffix]["reuse_params"] = reuse_from_name + suffix

    return f"{name}_res_add"


def add_mhsa_module(
    network: Dict,
    name: str,
    from_list: Union[str, List[str]],
    size: int = 512,
    num_att_heads: int = 8,
    clipping: int = 32,
    dropout: float = 0.1,
    l2: float = 5e-06,
    reuse_from_name: Optional[str] = None,
    mhsa_half_res_add: bool = False,
    **kwargs,
) -> str:
    network.update(
        {
            f"{name}_input": {
                "class": "copy",
                "from": from_list,
            },
            f"{name}_ln": {
                "class": "layer_norm",
                "from": f"{name}_input",
            },
            f"{name}_rel_pos_enc": {
                "class": "relative_positional_encoding",
                "from": f"{name}_ln",
                "n_out": size // num_att_heads,
                "clipping": clipping,
                "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_self_attention": {
                "class": "self_attention",
                "from": f"{name}_ln",
                "n_out": size,
                "num_heads": num_att_heads,
                "total_key_dim": size,
                "key_shift": f"{name}_rel_pos_enc",
                "attention_dropout": dropout,
                "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_att_linear": {
                "class": "linear",
                "activation": None,
                "from": f"{name}_self_attention",
                "n_out": size,
                "L2": l2,
                "with_bias": False,
                "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_dropout": {
                "class": "copy",
                "from": f"{name}_att_linear",
                "dropout": dropout,
            },
            f"{name}_res_add": {
                "from": [f"{name}_dropout", f"{name}_input"],
            },
        }
    )

    if mhsa_half_res_add:
        network[f"{name}_res_add"].update(
            {
                "class": "eval",
                "eval": "0.5 * source(0) + source(1)",
            }
        )
    else:
        network[f"{name}_res_add"].update(
            {
                "class": "combine",
                "kind": "add",
            }
        )

    if reuse_from_name:
        for suffix in ["_rel_pos_enc", "_self_attention", "_att_linear"]:
            network[name + suffix]["reuse_params"] = reuse_from_name + suffix

    return f"{name}_res_add"


def add_conv_module(
    network: Dict,
    name: str,
    from_list: Union[str, List[str]],
    size: int = 512,
    conv_filter_size: int = 32,
    dropout: float = 0.1,
    l2: float = 5e-06,
    reuse_from_name: Optional[str] = None,
    conv_half_res_add: bool = False,
    use_batch_norm: bool = True,
    **kwargs,
) -> str:
    network.update(
        {
            f"{name}_input": {
                "class": "copy",
                "from": from_list,
            },
            f"{name}_ln": {
                "class": "layer_norm",
                "from": f"{name}_input",
            },
            f"{name}_pointwise_conv_1": {
                "class": "linear",
                "activation": None,
                "from": f"{name}_ln",
                "n_out": size * 2,
                "L2": l2,
                "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_glu": {
                "class": "gating",
                "from": f"{name}_pointwise_conv_1",
                "activation": None,
                "gate_activation": "sigmoid",
            },
            f"{name}_depthwise_conv": {
                "class": "conv",
                "activation": None,
                "from": f"{name}_glu",
                "n_out": size,
                "filter_size": (conv_filter_size,),
                "padding": "same",
                "groups": size,
                "with_bias": True,
                "L2": l2,
                "forward_weights_init": get_variance_scaling_init(),
            },
        }
    )

    if use_batch_norm:
        norm_layer_name = f"{name}_bn"
        network[norm_layer_name] = {
            "class": "batch_norm",
            "from": f"{name}_depthwise_conv",
            "momentum": 0.1,
            "epsilon": 1e-05,
            "update_sample_only_in_training": True,
            "delay_sample_update": True,
        }
    else:
        norm_layer_name = f"{name}_replace_bn"
        network[norm_layer_name] = {
            "class": "layer_norm",
            "from": f"{name}_depthwise_conv",
        }

    network.update(
        {
            f"{name}_swish": {
                "class": "activation",
                "from": norm_layer_name,
                "activation": "swish",
            },
            f"{name}_pointwise_conv_2": {
                "class": "linear",
                "activation": None,
                "from": f"{name}_swish",
                "n_out": size,
                "L2": l2,
                "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_dropout": {
                "class": "copy",
                "from": f"{name}_pointwise_conv_2",
                "dropout": dropout,
            },
            f"{name}_res_add": {
                "from": [f"{name}_dropout", f"{name}_input"],
            },
        }
    )

    if conv_half_res_add:
        network[f"{name}_res_add"].update(
            {
                "class": "eval",
                "eval": "0.5 * source(0) + source(1)",
            }
        )
    else:
        network[f"{name}_res_add"].update(
            {
                "class": "combine",
                "kind": "add",
            }
        )

    if reuse_from_name:
        for suffix in ["_pointwise_conv_1", "_depthwise_conv", "_pointwise_conv_2"]:
            network[name + suffix]["reuse_params"] = reuse_from_name + suffix

    return f"{name}_res_add"


def add_conformer_block(
    network: Dict,
    name: str,
    from_list: Union[str, List[str]],
    reuse_from_name: Optional[str] = None,
    **kwargs,
) -> str:
    def ext_reuse_name(suffix) -> Optional[str]:
        if reuse_from_name is None:
            return None
        return reuse_from_name + suffix

    from_list = add_ff_module(
        network=network,
        name=f"{name}_ffmod_1",
        from_list=from_list,
        reuse_from_name=ext_reuse_name("_ffmod_1"),
        **kwargs,
    )
    from_list = add_conv_module(
        network=network,
        name=f"{name}_convmod_1",
        from_list=from_list,
        reuse_from_name=ext_reuse_name("_convmod"),
        **kwargs,
    )
    from_list = add_mhsa_module(
        network=network,
        name=f"{name}_mhsamod",
        from_list=from_list,
        reuse_from_name=ext_reuse_name("_mhsamod"),
        **kwargs,
    )
    from_list = add_ff_module(
        network=network,
        name=f"{name}_ffmod_2",
        from_list=from_list,
        reuse_from_name=ext_reuse_name("_ffmod_2"),
        **kwargs,
    )

    out_name = f"{name}_output"
    network[out_name] = {
        "class": "layer_norm",
        "from": from_list,
    }

    return out_name


def add_conformer_stack(
    network: Dict,
    from_list: Union[str, List[str]],
    name: str = "conformer",
    num_blocks: int = 12,
    reuse_from_name: Optional[str] = None,
    **kwargs,
) -> Tuple[str, List[str]]:
    block_names = []

    for idx in range(1, num_blocks + 1):
        layer_reuse_name = None
        if reuse_from_name:
            layer_reuse_name = f"{reuse_from_name}_{idx}"
        from_list = add_conformer_block(
            network,
            name=f"{name}_{idx}",
            from_list=from_list,
            reuse_from_name=layer_reuse_name,
            **kwargs,
        )
        block_names.append(from_list)

    return from_list, block_names


def add_initial_conv(
    network: Dict,
    from_list: Union[str, List[str]],
    name: str = "vgg_frontend",
    linear_size: int = 512,
    conv_outputs: List[int] = [32, 64, 64],
    conv_filters: List[int] = [3, 3, 3],
    conv_strides: List[int] = [1, 2, 2],
    max_pool: List[int] = [2],
    reuse_from_name: Optional[str] = None,
    dropout: float = 0.1,
    **kwargs,
) -> str:
    from_name = f"{name}_split_dims"
    network[from_name] = {
        "class": "split_dims",
        "from": from_list,
        "dims": (-1, 1),
        "axis": "F",
    }

    for idx, (n_out, filter, stride) in enumerate(zip(conv_outputs, conv_filters, conv_strides), start=1):
        network[f"{name}_conv_{idx}"] = {
            "class": "conv",
            "activation": "swish",
            "from": from_name,
            "n_out": n_out,
            "filter_size": (filter, filter),
            "with_bias": True,
            "padding": "same",
            "L2": 0.01,
        }
        if stride != 1:
            network[f"{name}_conv_{idx}"]["strides"] = (stride, 1)

        from_name = f"{name}_conv_{idx}"

        if len(max_pool) >= idx and max_pool[idx - 1] != 1:
            network[f"{name}_pool_{idx}"] = {
                "class": "pool",
                "from": from_name,
                "mode": "max",
                "pool_size": (1, max_pool[idx - 1]),
                "padding": "same",
                "trainable": False,
            }
            from_name = f"{name}_pool_{idx}"

    network[f"{name}_merge_dims"] = {
        "class": "merge_dims",
        "from": from_name,
        "axes": "static",
    }

    network[f"{name}_linear"] = {
        "class": "linear",
        "activation": None,
        "from": f"{name}_merge_dims",
        "n_out": linear_size,
        "with_bias": False,
        "L2": 1e-04,
    }

    network[f"{name}_dropout"] = {
        "class": "copy",
        "from": f"{name}_linear",
        "dropout": dropout,
    }

    if reuse_from_name:
        for suffix in [f"_conv_{idx}" for idx in range(1, len(conv_outputs) + 1)] + ["_linear"]:
            network[name + suffix]["reuse_params"] = reuse_from_name + suffix

    return f"{name}_dropout"
