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
            'from': ['source'] }
    })
    return net, "source0"

def make_final_output(
    net = None,
    in_l = None
):
    net.update({
        'length_masked': {
            'class': 'reinterpret_data', 
            'from': [in_l], 
            'size_base': 
            'data:classes'},
        'output': { 
            'class': 'softmax',
            'dropout': 0.05,
            'from': ['length_masked'],
            'loss': 'ce',
            'loss_opts': {
                'focal_loss_factor': 0.0, 
                'label_smoothing': 0.0, 
                'use_normalized_loss': False},
            'target': 'classes'},
    })

    return net, "output"


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
    return filtered

def pprint(data):
    import yaml
    print(yaml.dump(data, default_flow_style=False))


from .conv_mod_versions import make_conv_mod_001
from .subsampling_versions import make_subsampling_001, make_unsampling_001
from .ff_mod_versions import make_ff_mod_001
from .sa_mod_versions import make_self_att_mod_001


def make_conformer_baseline(

    subsampling_func=make_subsampling_001,
    unsampling_func=make_unsampling_001,
    sampling_func_args=None,

    conformer_ff1_func=make_ff_mod_001,
    ff1_func_args=None,

    conformer_self_att_func=make_self_att_mod_001,
    sa_func_args=None,

    conformer_self_conv_func=make_conv_mod_001,
    conv_func_args=None,

    conformer_ff2_func=make_ff_mod_001,
    ff2_func_args=None,

    shared_model_args=None,

    num_blocks = None,

    print_net = False
):
    net, last = {}, "data"
    net, last = make_inital_transformations(net, last)
    net, last = subsampling_func(
        net, last,
        **sampling_func_args,
        **filter_args_for_func(subsampling_func, shared_model_args)
    )

    for i in range(num_blocks):
        block_str = f"enc_{i:03d}"

        # FF1
        net, last = conformer_ff1_func(
            net, last,
            prefix = f"{block_str}_ff1",
            **ff1_func_args,
            **filter_args_for_func(conformer_ff1_func, shared_model_args)
        )

        # SA
        net, last = conformer_self_att_func(
            net, last,
            prefix = f"{block_str}",
            **sa_func_args,
            **filter_args_for_func(conformer_self_att_func, shared_model_args)
        )
        
        # CONV MOD
        net, last = conformer_self_conv_func(
            net, last,
            prefix = f"{block_str}",
            **conv_func_args,
            **filter_args_for_func(conformer_self_conv_func, shared_model_args)
        )

        # FF2
        net, last = conformer_ff2_func(
            net, last,
            prefix = f"{block_str}_ff2",
            **ff2_func_args,
            **filter_args_for_func(conformer_ff1_func, shared_model_args)
        )


    # This baseline needs no unsampling, because it has no downsampling
    net, last = make_final_output(net, last)

    if print_net:
        pprint(net) 

    return net


def make_conformer_00(

    subsampling_func=make_subsampling_001,
    unsampling_func=make_unsampling_001,
    sampling_func_args=None,

    conformer_ff1_func=make_ff_mod_001,
    ff1_func_args=None,

    conformer_self_att_func=make_self_att_mod_001,
    sa_func_args=None,

    conformer_self_conv_func=make_conv_mod_001,
    conv_func_args=None,

    conformer_ff2_func=make_ff_mod_001,
    ff2_func_args=None,

    shared_model_args=None,

    num_blocks = None,

    print_net = False
):
    net, last = {}, "data"
    net, last = make_inital_transformations(net, last)
    net, last = subsampling_func(
        net, last,
        **sampling_func_args,
        **filter_args_for_func(subsampling_func, shared_model_args)
    )

    for i in range(num_blocks):
        block_str = f"enc_{i:03d}"

        # FF1
        net, last = conformer_ff1_func(
            net, last,
            prefix = f"{block_str}_ff1",
            **ff1_func_args,
            **filter_args_for_func(conformer_ff1_func, shared_model_args)
        )

        # SA
        net, last = conformer_self_att_func(
            net, last,
            prefix = f"{block_str}",
            **sa_func_args,
            **filter_args_for_func(conformer_self_att_func, shared_model_args)
        )
        
        # CONV MOD
        net, last = conformer_self_conv_func(
            net, last,
            prefix = f"{block_str}",
            **conv_func_args,
            **filter_args_for_func(conformer_self_conv_func, shared_model_args)
        )

        # FF2
        net, last = conformer_ff2_func(
            net, last,
            prefix = f"{block_str}_ff2",
            **ff2_func_args,
            **filter_args_for_func(conformer_ff1_func, shared_model_args)
        )

    net, last = unsampling_func(
        net, last,
        # We also filter these cause not all ars of un- and sub- must match
        **filter_args_for_func(unsampling_func, sampling_func_args),
        **filter_args_for_func(unsampling_func, shared_model_args)
    )

    net, last = make_final_output(net, last)

    if print_net:
        pprint(net) 

    return net
