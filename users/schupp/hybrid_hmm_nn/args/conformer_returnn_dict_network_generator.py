# Oly style returnn dict network generator
from typing import Tuple, List, Callable, Optional, Union, Any

def prefix_all_keys(
    prefix = None,
    net = None,
    in_l = None
):
    # All all prefixes to keys
    prefixed = {}
    for x in net.keys():
        prefixed[prefix + x] = net[x]

    # Maybe add prefixed to 'from'
    for x in prefixed.keys():
        if prefixed[x]["from"] != in_l:
            if isinstance(prefixed[x]["from"], list):
                for i in range(len(prefixed[x]["from"])):
                    if prefixed[x]["from"][i] != in_l:
                        prefixed[x]["from"][i] = prefix + prefixed[x]["from"][i]
            else:
                prefixed[x]["from"] = prefix + prefixed[x]["from"]

    return prefixed

def make_subsampling_001(
    net=None,
    in_l="source0",
    time_reduction=1
):
    assert net, "need network"
    net.update({
        "conv0_0" : {
            "class": "conv", 
            "from": in_l, 
            "padding": "same",
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": None, 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0_1" : {
            "class": "conv", 
            "from": f"conv0_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 32, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:50"]},  # (T,50,32)
        "conv0p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (1, 2), 
            'strides': (1, 2),
            "from": "conv0_1", 
            "in_spatial_dims": ["T", "dim:50"] }, # (T, 25, 32)
        "conv1_0" : {
            "class": "conv", 
            "from": "conv0p", 
            "padding": "same", 
            "filter_size": (3, 3), 
            "n_out": 64,
            "activation": None, 
            "with_bias": True,
            "in_spatial_dims": ["T", "dim:25"] }, # (T, 25, 64)
        "conv1_1" : {
            "class": "conv", 
            "from": "conv1_0", 
            "padding": "same", 
            "filter_size": (3, 3),
            "n_out": 64, 
            "activation": 'relu', 
            "with_bias": True, 
            "in_spatial_dims": ["T", "dim:25"]}, # (T,25,64)
        "conv1p" : {
            "class": "pool", 
            "mode": "max", 
            "padding": "same", 
            "pool_size": (time_reduction, 1), 
            'strides': (time_reduction, 1),
            "from": "conv1_1", 
            "in_spatial_dims": ["T", "dim:25"]},
        "conv_merged" : {
            "class": "merge_dims", 
            "from": "conv1p", 
            "axes": ["dim:25", "dim:64"]}
    })
    return net, "conv_merged"

def make_inital_transformations(
    net = {},
    in_l = "data"
) -> Tuple[dict, str]:

    net.update({
        "source" : {
            'class': 'eval', 
            'eval': "self.network.get_config().typed_value('transform')(source(0), network=self.network)", 
            'from': in_l},
        "source0" : {
            "class": "split_dims", 
            "axis": "F", 
            "dims": (-1, 1), 
            'from': ['source0'] }
    })
    return net, "source0"

def make_ff_mod_001(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_initialization = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,

    # Model shared args
    model_dim = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    net_add = {
        "_laynorm" : {
            'class': "layer_norm",
            'from': in_l},
        "_conv1" : {
            'class': "linear", 
            'activation': ff_activation,
            'with_bias': True,
            'from': ["_laynorm"],
            'n_out': ff_dim, 
            'forward_weights_init': ff_initialization },
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_conv1"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': ff_initialization},
        "_drop" : {
            'class': "dropout",
            'dropout': ff_post_dropout,
            'from': ["_conv2"]},
        "_drop_half" : {
            'class': "eval",
            'eval': f"{ff_half_ratio} * source(0)",
            'from': ["_drop"] },
        "_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_drop_half"]
        }
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_out"

def filter_args_for_func(
    func,
    args
):
    import inspect

    sig = inspect.signature(func)
    params = [p.name for p in sig.parameters.values()]
    filtered = {}
    for x in args.keys():
        if x in params:
            filtered[x] = args[x]
    return args

def pprint(data):
    import yaml
    print(yaml.dump(data, default_flow_style=False))

def make_conformer_00(

    subsampling_func=make_subsampling_001,

    conformer_ff1_func=make_ff_mod_001,
    ff1_func_args=None,

#   conformer_self_att_func=make_self_att_mod_001,
#   conformer_self_conv_func=make_conv_mod_001,
    conformer_ff2_func=make_ff_mod_001,
    ff2_func_args=None,

    shared_model_args=None,

    num_blocks = 1,
):
    net, last = {}, "data"
    net, last = make_inital_transformations(net, last)
    net, last = subsampling_func(net, last)

    for i in range(num_blocks):
        block_str = f"enc_{i:03d}"
        net, last = conformer_ff1_func(
            net, last,
            prefix = f"{block_str}_ff1",
            **ff1_func_args,
            **filter_args_for_func(conformer_ff1_func, shared_model_args)
        )

        #net, last = conformer_self_att_func(net, last)
        #net, last = conformer_self_conv_func(net, last)
        net, last = conformer_ff2_func(
            net, last,
            prefix = f"{block_str}_ff2",
            **ff2_func_args
        )


    pprint(net) 
    #net, last = make_final_output(net, last)
