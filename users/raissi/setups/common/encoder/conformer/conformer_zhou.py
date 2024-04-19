__all__ = ["add_conformer_stack"]

from typing import Any, Dict, List, Optional, Tuple, Union

import i6_core.returnn as returnn


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
                # "forward_weights_init": get_variance_scaling_init(),
                "L2": l2,
            },
            f"{name}_ff_2": {
                "class": "linear",
                "activation": None,
                "from": f"{name}_ff_1",
                "n_out": size,
                "L2": l2,
                # "forward_weights_init": get_variance_scaling_init(),
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
                # "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_self_attention": {
                "class": "self_attention",
                "from": f"{name}_ln",
                "n_out": size,
                "num_heads": num_att_heads,
                "total_key_dim": size,
                "key_shift": f"{name}_rel_pos_enc",
                "attention_dropout": dropout,
                # "forward_weights_init": get_variance_scaling_init(),
            },
            f"{name}_att_linear": {
                "class": "linear",
                "activation": None,
                "from": f"{name}_self_attention",
                "n_out": size,
                "L2": l2,
                "with_bias": False,
                # "forward_weights_init": get_variance_scaling_init(),
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
                # "forward_weights_init": get_variance_scaling_init(),
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
                # "forward_weights_init": get_variance_scaling_init(),
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
                # "forward_weights_init": get_variance_scaling_init(),
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


def get_conformer_network_moritz_dict(
    conf_model_dim: int,
    l2: float = 5e-6,
    ss_factor: int = 4,
    as_data_specaugment: bool = True,
    out_layer_name: str = "encoder-output",
) -> returnn.ReturnnConfig:
    assert ss_factor == 4, "unimplemented"

    network = {
        "input_dropout": {"class": "copy", "dropout": 0.1, "from": "input_linear"},
        "input_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conv_merged",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "source": {
            "class": "eval",
            "from": "data",
            "eval": f"self.network.get_config().typed_value('transform')(source(0, as_data={as_data_specaugment}), network=self.network)",
        },
        "conv_1": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_source",
            "n_out": 32,
            "padding": "same",
            "with_bias": True,
        },
        "conv_1_pool": {
            "class": "pool",
            "from": "conv_1",
            "mode": "max",
            "padding": "same",
            "pool_size": (1, 2),
            "trainable": False,
        },
        "conv_2": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_1_pool",
            "n_out": 64,
            "padding": "same",
            "strides": (2, 1),
            "with_bias": True,
        },
        "conv_3": {
            "L2": 0.01,
            "activation": "swish",
            "class": "conv",
            "filter_size": (3, 3),
            "from": "conv_2",
            "n_out": 64,
            "padding": "same",
            "strides": (2, 1),
            "with_bias": True,
        },
        "conv_merged": {"axes": "static", "class": "merge_dims", "from": "conv_3"},
        "conv_source": {"axis": "F", "class": "split_dims", "dims": (-1, 1), "from": "source"},
        "conformer_01_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_01_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_01_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_01_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_01_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_conv_mod_pointwise_conv_2",
        },
        "conformer_01_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_01_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_01_conv_mod_ln": {"class": "layer_norm", "from": "conformer_01_ffmod_1_half_res_add"},
        "conformer_01_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_01_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_01_conv_mod_dropout", "conformer_01_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_01_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_01_conv_mod_bn",
        },
        "conformer_01_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_1_dropout_linear",
        },
        "conformer_01_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_01_ffmod_1_dropout", "input_dropout"],
        },
        "conformer_01_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_01_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_01_ffmod_1_ln": {"class": "layer_norm", "from": "input_dropout"},
        "conformer_01_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_2_dropout_linear",
        },
        "conformer_01_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_01_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_01_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_01_ffmod_2_dropout", "conformer_01_mhsa_mod_res_add"],
        },
        "conformer_01_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_01_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_01_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_01_mhsa_mod_res_add"},
        "conformer_01_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_01_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_01_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_01_mhsa_mod_att_linear"},
        "conformer_01_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_01_conv_mod_res_add"},
        "conformer_01_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_01_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_01_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_01_mhsa_mod_dropout", "conformer_01_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_01_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_01_mhsa_mod_ln",
            "key_shift": "conformer_01_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_01_output": {"class": "layer_norm", "from": "conformer_01_ffmod_2_half_res_add"},
        "conformer_02_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_02_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_02_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_02_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_02_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_conv_mod_pointwise_conv_2",
        },
        "conformer_02_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_02_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_02_conv_mod_ln": {"class": "layer_norm", "from": "conformer_02_ffmod_1_half_res_add"},
        "conformer_02_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_02_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_02_conv_mod_dropout", "conformer_02_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_02_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_02_conv_mod_bn",
        },
        "conformer_02_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_1_dropout_linear",
        },
        "conformer_02_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_02_ffmod_1_dropout", "conformer_01_output"],
        },
        "conformer_02_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_02_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_02_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_01_output"},
        "conformer_02_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_2_dropout_linear",
        },
        "conformer_02_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_02_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_02_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_02_ffmod_2_dropout", "conformer_02_mhsa_mod_res_add"],
        },
        "conformer_02_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_02_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_02_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_02_mhsa_mod_res_add"},
        "conformer_02_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_02_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_02_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_02_mhsa_mod_att_linear"},
        "conformer_02_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_02_conv_mod_res_add"},
        "conformer_02_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_02_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_02_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_02_mhsa_mod_dropout", "conformer_02_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_02_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_02_mhsa_mod_ln",
            "key_shift": "conformer_02_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_02_output": {"class": "layer_norm", "from": "conformer_02_ffmod_2_half_res_add"},
        "conformer_03_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_03_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_03_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_03_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_03_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_conv_mod_pointwise_conv_2",
        },
        "conformer_03_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_03_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_03_conv_mod_ln": {"class": "layer_norm", "from": "conformer_03_ffmod_1_half_res_add"},
        "conformer_03_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_03_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_03_conv_mod_dropout", "conformer_03_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_03_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_03_conv_mod_bn",
        },
        "conformer_03_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_1_dropout_linear",
        },
        "conformer_03_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_03_ffmod_1_dropout", "conformer_02_output"],
        },
        "conformer_03_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_03_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_03_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_02_output"},
        "conformer_03_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_2_dropout_linear",
        },
        "conformer_03_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_03_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_03_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_03_ffmod_2_dropout", "conformer_03_mhsa_mod_res_add"],
        },
        "conformer_03_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_03_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_03_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_03_mhsa_mod_res_add"},
        "conformer_03_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_03_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_03_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_03_mhsa_mod_att_linear"},
        "conformer_03_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_03_conv_mod_res_add"},
        "conformer_03_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_03_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_03_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_03_mhsa_mod_dropout", "conformer_03_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_03_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_03_mhsa_mod_ln",
            "key_shift": "conformer_03_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_03_output": {"class": "layer_norm", "from": "conformer_03_ffmod_2_half_res_add"},
        "conformer_04_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_04_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_04_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_04_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_04_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_conv_mod_pointwise_conv_2",
        },
        "conformer_04_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_04_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_04_conv_mod_ln": {"class": "layer_norm", "from": "conformer_04_ffmod_1_half_res_add"},
        "conformer_04_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_04_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_04_conv_mod_dropout", "conformer_04_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_04_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_04_conv_mod_bn",
        },
        "conformer_04_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_1_dropout_linear",
        },
        "conformer_04_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_04_ffmod_1_dropout", "conformer_03_output"],
        },
        "conformer_04_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_04_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_04_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_03_output"},
        "conformer_04_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_2_dropout_linear",
        },
        "conformer_04_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_04_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_04_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_04_ffmod_2_dropout", "conformer_04_mhsa_mod_res_add"],
        },
        "conformer_04_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_04_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_04_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_04_mhsa_mod_res_add"},
        "conformer_04_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_04_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_04_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_04_mhsa_mod_att_linear"},
        "conformer_04_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_04_conv_mod_res_add"},
        "conformer_04_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_04_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_04_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_04_mhsa_mod_dropout", "conformer_04_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_04_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_04_mhsa_mod_ln",
            "key_shift": "conformer_04_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_04_output": {"class": "layer_norm", "from": "conformer_04_ffmod_2_half_res_add"},
        "conformer_05_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_05_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_05_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_05_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_05_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_conv_mod_pointwise_conv_2",
        },
        "conformer_05_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_05_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_05_conv_mod_ln": {"class": "layer_norm", "from": "conformer_05_ffmod_1_half_res_add"},
        "conformer_05_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_05_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_05_conv_mod_dropout", "conformer_05_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_05_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_05_conv_mod_bn",
        },
        "conformer_05_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_1_dropout_linear",
        },
        "conformer_05_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_05_ffmod_1_dropout", "conformer_04_output"],
        },
        "conformer_05_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_05_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_05_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_04_output"},
        "conformer_05_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_2_dropout_linear",
        },
        "conformer_05_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_05_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_05_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_05_ffmod_2_dropout", "conformer_05_mhsa_mod_res_add"],
        },
        "conformer_05_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_05_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_05_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_05_mhsa_mod_res_add"},
        "conformer_05_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_05_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_05_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_05_mhsa_mod_att_linear"},
        "conformer_05_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_05_conv_mod_res_add"},
        "conformer_05_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_05_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_05_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_05_mhsa_mod_dropout", "conformer_05_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_05_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_05_mhsa_mod_ln",
            "key_shift": "conformer_05_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_05_output": {"class": "layer_norm", "from": "conformer_05_ffmod_2_half_res_add"},
        "conformer_06_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_06_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_06_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_06_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_06_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_conv_mod_pointwise_conv_2",
        },
        "conformer_06_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_06_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_06_conv_mod_ln": {"class": "layer_norm", "from": "conformer_06_ffmod_1_half_res_add"},
        "conformer_06_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_06_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_06_conv_mod_dropout", "conformer_06_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_06_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_06_conv_mod_bn",
        },
        "conformer_06_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_1_dropout_linear",
        },
        "conformer_06_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_06_ffmod_1_dropout", "conformer_05_output"],
        },
        "conformer_06_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_06_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_06_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_05_output"},
        "conformer_06_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_2_dropout_linear",
        },
        "conformer_06_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_06_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_06_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_06_ffmod_2_dropout", "conformer_06_mhsa_mod_res_add"],
        },
        "conformer_06_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_06_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_06_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_06_mhsa_mod_res_add"},
        "conformer_06_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_06_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_06_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_06_mhsa_mod_att_linear"},
        "conformer_06_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_06_conv_mod_res_add"},
        "conformer_06_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_06_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_06_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_06_mhsa_mod_dropout", "conformer_06_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_06_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_06_mhsa_mod_ln",
            "key_shift": "conformer_06_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_06_output": {"class": "layer_norm", "from": "conformer_06_ffmod_2_half_res_add"},
        "conformer_07_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_07_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_07_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_07_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_07_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_conv_mod_pointwise_conv_2",
        },
        "conformer_07_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_07_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_07_conv_mod_ln": {"class": "layer_norm", "from": "conformer_07_ffmod_1_half_res_add"},
        "conformer_07_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_07_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_07_conv_mod_dropout", "conformer_07_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_07_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_07_conv_mod_bn",
        },
        "conformer_07_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_1_dropout_linear",
        },
        "conformer_07_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_07_ffmod_1_dropout", "conformer_06_output"],
        },
        "conformer_07_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_07_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_07_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_06_output"},
        "conformer_07_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_2_dropout_linear",
        },
        "conformer_07_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_07_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_07_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_07_ffmod_2_dropout", "conformer_07_mhsa_mod_res_add"],
        },
        "conformer_07_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_07_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_07_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_07_mhsa_mod_res_add"},
        "conformer_07_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_07_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_07_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_07_mhsa_mod_att_linear"},
        "conformer_07_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_07_conv_mod_res_add"},
        "conformer_07_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_07_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_07_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_07_mhsa_mod_dropout", "conformer_07_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_07_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_07_mhsa_mod_ln",
            "key_shift": "conformer_07_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_07_output": {"class": "layer_norm", "from": "conformer_07_ffmod_2_half_res_add"},
        "conformer_08_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_08_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_08_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_08_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_08_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_conv_mod_pointwise_conv_2",
        },
        "conformer_08_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_08_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_08_conv_mod_ln": {"class": "layer_norm", "from": "conformer_08_ffmod_1_half_res_add"},
        "conformer_08_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_08_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_08_conv_mod_dropout", "conformer_08_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_08_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_08_conv_mod_bn",
        },
        "conformer_08_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_1_dropout_linear",
        },
        "conformer_08_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_08_ffmod_1_dropout", "conformer_07_output"],
        },
        "conformer_08_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_08_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_08_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_07_output"},
        "conformer_08_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_2_dropout_linear",
        },
        "conformer_08_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_08_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_08_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_08_ffmod_2_dropout", "conformer_08_mhsa_mod_res_add"],
        },
        "conformer_08_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_08_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_08_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_08_mhsa_mod_res_add"},
        "conformer_08_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_08_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_08_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_08_mhsa_mod_att_linear"},
        "conformer_08_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_08_conv_mod_res_add"},
        "conformer_08_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_08_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_08_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_08_mhsa_mod_dropout", "conformer_08_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_08_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_08_mhsa_mod_ln",
            "key_shift": "conformer_08_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_08_output": {"class": "layer_norm", "from": "conformer_08_ffmod_2_half_res_add"},
        "conformer_09_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_09_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_09_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_09_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_09_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_conv_mod_pointwise_conv_2",
        },
        "conformer_09_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_09_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_09_conv_mod_ln": {"class": "layer_norm", "from": "conformer_09_ffmod_1_half_res_add"},
        "conformer_09_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_09_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_09_conv_mod_dropout", "conformer_09_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_09_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_09_conv_mod_bn",
        },
        "conformer_09_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_1_dropout_linear",
        },
        "conformer_09_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_09_ffmod_1_dropout", "conformer_08_output"],
        },
        "conformer_09_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_09_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_09_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_08_output"},
        "conformer_09_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_2_dropout_linear",
        },
        "conformer_09_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_09_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_09_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_09_ffmod_2_dropout", "conformer_09_mhsa_mod_res_add"],
        },
        "conformer_09_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_09_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_09_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_09_mhsa_mod_res_add"},
        "conformer_09_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_09_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_09_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_09_mhsa_mod_att_linear"},
        "conformer_09_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_09_conv_mod_res_add"},
        "conformer_09_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_09_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_09_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_09_mhsa_mod_dropout", "conformer_09_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_09_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_09_mhsa_mod_ln",
            "key_shift": "conformer_09_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_09_output": {"class": "layer_norm", "from": "conformer_09_ffmod_2_half_res_add"},
        "conformer_10_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_10_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_10_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_10_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_10_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_conv_mod_pointwise_conv_2",
        },
        "conformer_10_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_10_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_10_conv_mod_ln": {"class": "layer_norm", "from": "conformer_10_ffmod_1_half_res_add"},
        "conformer_10_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_10_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_10_conv_mod_dropout", "conformer_10_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_10_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_10_conv_mod_bn",
        },
        "conformer_10_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_1_dropout_linear",
        },
        "conformer_10_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_10_ffmod_1_dropout", "conformer_09_output"],
        },
        "conformer_10_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_10_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_10_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_09_output"},
        "conformer_10_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_2_dropout_linear",
        },
        "conformer_10_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_10_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_10_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_10_ffmod_2_dropout", "conformer_10_mhsa_mod_res_add"],
        },
        "conformer_10_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_10_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_10_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_10_mhsa_mod_res_add"},
        "conformer_10_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_10_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_10_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_10_mhsa_mod_att_linear"},
        "conformer_10_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_10_conv_mod_res_add"},
        "conformer_10_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_10_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_10_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_10_mhsa_mod_dropout", "conformer_10_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_10_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_10_mhsa_mod_ln",
            "key_shift": "conformer_10_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_10_output": {"class": "layer_norm", "from": "conformer_10_ffmod_2_half_res_add"},
        "conformer_11_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_11_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_11_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_11_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_11_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_conv_mod_pointwise_conv_2",
        },
        "conformer_11_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_11_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_11_conv_mod_ln": {"class": "layer_norm", "from": "conformer_11_ffmod_1_half_res_add"},
        "conformer_11_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_11_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_11_conv_mod_dropout", "conformer_11_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_11_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_11_conv_mod_bn",
        },
        "conformer_11_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_1_dropout_linear",
        },
        "conformer_11_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_11_ffmod_1_dropout", "conformer_10_output"],
        },
        "conformer_11_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_11_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_11_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_10_output"},
        "conformer_11_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_2_dropout_linear",
        },
        "conformer_11_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_11_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_11_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_11_ffmod_2_dropout", "conformer_11_mhsa_mod_res_add"],
        },
        "conformer_11_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_11_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_11_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_11_mhsa_mod_res_add"},
        "conformer_11_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_11_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_11_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_11_mhsa_mod_att_linear"},
        "conformer_11_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_11_conv_mod_res_add"},
        "conformer_11_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_11_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_11_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_11_mhsa_mod_dropout", "conformer_11_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_11_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_11_mhsa_mod_ln",
            "key_shift": "conformer_11_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_11_output": {"class": "layer_norm", "from": "conformer_11_ffmod_2_half_res_add"},
        "conformer_12_conv_mod_bn": {
            "class": "batch_norm",
            "delay_sample_update": True,
            "epsilon": 1e-05,
            "from": "conformer_12_conv_mod_depthwise_conv",
            "momentum": 0.1,
            "update_sample_only_in_training": True,
        },
        "conformer_12_conv_mod_depthwise_conv": {
            "L2": l2,
            "activation": None,
            "class": "conv",
            "filter_size": (32,),
            "from": "conformer_12_conv_mod_glu",
            "groups": conf_model_dim,
            "n_out": conf_model_dim,
            "padding": "same",
            "with_bias": True,
        },
        "conformer_12_conv_mod_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_conv_mod_pointwise_conv_2",
        },
        "conformer_12_conv_mod_glu": {
            "activation": None,
            "class": "gating",
            "from": "conformer_12_conv_mod_pointwise_conv_1",
            "gate_activation": "sigmoid",
        },
        "conformer_12_conv_mod_ln": {"class": "layer_norm", "from": "conformer_12_ffmod_1_half_res_add"},
        "conformer_12_conv_mod_pointwise_conv_1": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_conv_mod_ln",
            "n_out": 2 * conf_model_dim,
        },
        "conformer_12_conv_mod_pointwise_conv_2": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_conv_mod_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_conv_mod_res_add": {
            "class": "combine",
            "from": ["conformer_12_conv_mod_dropout", "conformer_12_ffmod_1_half_res_add"],
            "kind": "add",
        },
        "conformer_12_conv_mod_swish": {
            "activation": "swish",
            "class": "activation",
            "from": "conformer_12_conv_mod_bn",
        },
        "conformer_12_ffmod_1_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_1_dropout_linear",
        },
        "conformer_12_ffmod_1_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_1_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_ffmod_1_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_12_ffmod_1_dropout", "conformer_11_output"],
        },
        "conformer_12_ffmod_1_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_12_ffmod_1_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_12_ffmod_1_ln": {"class": "layer_norm", "from": "conformer_11_output"},
        "conformer_12_ffmod_2_dropout": {
            "class": "copy",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_2_dropout_linear",
        },
        "conformer_12_ffmod_2_dropout_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "dropout": 0.1,
            "from": "conformer_12_ffmod_2_linear_swish",
            "n_out": conf_model_dim,
        },
        "conformer_12_ffmod_2_half_res_add": {
            "class": "eval",
            "eval": "0.5 * source(0) + source(1)",
            "from": ["conformer_12_ffmod_2_dropout", "conformer_12_mhsa_mod_res_add"],
        },
        "conformer_12_ffmod_2_linear_swish": {
            "L2": l2,
            "activation": "swish",
            "class": "linear",
            "from": "conformer_12_ffmod_2_ln",
            "n_out": 4 * conf_model_dim,
        },
        "conformer_12_ffmod_2_ln": {"class": "layer_norm", "from": "conformer_12_mhsa_mod_res_add"},
        "conformer_12_mhsa_mod_att_linear": {
            "L2": l2,
            "activation": None,
            "class": "linear",
            "from": "conformer_12_mhsa_mod_self_attention",
            "n_out": conf_model_dim,
            "with_bias": False,
        },
        "conformer_12_mhsa_mod_dropout": {"class": "copy", "dropout": 0.1, "from": "conformer_12_mhsa_mod_att_linear"},
        "conformer_12_mhsa_mod_ln": {"class": "layer_norm", "from": "conformer_12_conv_mod_res_add"},
        "conformer_12_mhsa_mod_relpos_encoding": {
            "class": "relative_positional_encoding",
            "clipping": 32,
            "from": "conformer_12_mhsa_mod_ln",
            "n_out": 64,
        },
        "conformer_12_mhsa_mod_res_add": {
            "class": "combine",
            "from": ["conformer_12_mhsa_mod_dropout", "conformer_12_conv_mod_res_add"],
            "kind": "add",
        },
        "conformer_12_mhsa_mod_self_attention": {
            "attention_dropout": 0.1,
            "class": "self_attention",
            "from": "conformer_12_mhsa_mod_ln",
            "key_shift": "conformer_12_mhsa_mod_relpos_encoding",
            "n_out": conf_model_dim,
            "num_heads": 8,
            "total_key_dim": conf_model_dim,
        },
        "conformer_12_output": {"class": "layer_norm", "from": "conformer_12_ffmod_2_half_res_add"},
        "enc_006": {  # for aux loss
            "class": "copy",
            "from": "conformer_06_output",
            "n_out": conf_model_dim,
        },
        out_layer_name: {
            "class": "copy",
            "from": "conformer_12_output",
            "n_out": conf_model_dim,
        },
    }

    return network
