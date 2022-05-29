from typing import OrderedDict
from .conformer_returnn_dict_network_generator import prefix_all_keys

def make_conv_mod_001(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_batchnorm" : {
            'class': "batch_norm",
            'masked_time' : True,
            'from': ["_conv_depthwise"]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"

def make_conv_mod_002( # This version has NO batch norm
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_depthwise"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"

def make_conv_mod_003_sd( # Version has stochastic depth
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,
    survival_prob = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):

    from .network_additions import stochatic_depth_00
    subnet = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
#        "_conv_batchnorm" : { # For some weird reason batch_norm gives complication when used in subnet, this thows:  'batch_norm/.../conv .. has been marked as not fetchable'
#            'class': "batch_norm",
#            'masked_time' : True,
#            'from': ["_conv_depthwise"]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
#            'from': ["_conv_batchnorm"]},
            'from': ["_conv_depthwise"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
    }

    prefixed_subnet = prefix_all_keys(prefix, subnet, in_l)

    net_sd, out = stochatic_depth_00(
        subnetwork=prefixed_subnet,
        survival_prob=survival_prob,
        subnet_last=f"{prefix}_conv_dropout",
        in_l = in_l,

        prefix=f"{prefix}_conv"
    )

    net.update(net_sd)

    return net, out

def make_conv_mod_004_layer_norm(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_batchnorm" : {
            'class': "layer_norm",
            'from': ["_conv_depthwise"]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"


def make_conv_mod_004_old_defaults_batchnorm_or_dynamic(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    batch_norm_settings = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):

    if batch_norm_settings is None:
        # if not specified, use old defaults as per: https://returnn.readthedocs.io/en/latest/configuration_reference/behavior_version.html?highlight=behavior_version
        batch_norm_settings = OrderedDict(
            momentum = 0.1,
            update_sample_only_in_training = False,
            delay_sample_update = False,
            param_version = 0,
            masked_time = True,
        )

    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_batchnorm" : {
            'class': "batch_norm",
            'from': ["_conv_depthwise"],
            **batch_norm_settings},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"


def make_conv_mod_006_se_block( # TODO
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,
    use_se_block = False,
    se_version = 0,
    se_dim = None,
    se_act = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):

    from .network_additions import se_block, se_block_scale
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True}
    }

    next_layer = "_conv_depthwise"

    if use_se_block:
        if se_version == 0:
            net_add, next_layer = se_block(net_add, next_layer, prefix="_conv", model_dim=model_dim)
        elif se_version == 1:
            if not se_act:
                net_add, next_layer = se_block_scale(net_add, next_layer, prefix="_conv", model_dim=model_dim, se_dim=se_dim)
            else:
                net_add, next_layer = se_block_scale(net_add, next_layer, prefix="_conv", model_dim=model_dim, se_dim=se_dim, se_act=se_act)

    net_add2 = {
        "_conv_batchnorm" : {
            'class': "layer_norm",
            'from': [next_layer]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )


    return net, f"{prefix}_conv_output"


def make_conv_mod_007_group_norm(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,
    group_norm_axes = "TF",

    # Shared network args:
    initialization = None,
    model_dim = None,

):
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_batchnorm" : {
            'class': "norm",
            'axes' : group_norm_axes,
            'from': ["_conv_depthwise"]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"

def make_conv_mod_008_group_norm_custom(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    groups = None,
    epsilon = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):

    from .network_additions import group_normalization
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True}
    }

    net_add, _next = group_normalization(net_add, "_conv_depthwise", "", groups, epsilon, model_dim)

    net_add2 = {
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': [_next]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"


def make_conv_mod_009_group_norm_custom_tf(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    groups = None,
    epsilon = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):

    from .network_additions import group_normalization
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_groupnorm": {
            "class" : "eval",
            "from" : "_conv_depthwise",
            "eval" : f"self.network.get_config().typed_value('tf_group_norm')(source(0), G={groups}, eps={epsilon})"},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': "_conv_groupnorm" },
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"


def make_conv_mod_010_initial_groupnorm(
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,

    # Groupnorm settings:
    groups = None,
    epsilon = None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):


    from .network_additions import group_normalization

    net_add = {}
    net_add, _next = group_normalization(net_add, in_l, "_conv", groups, epsilon, model_dim)

    net_add.update({
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': [_next], 
            'n_out': 2 * model_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True},
        "_conv_batchnorm" : {
            'class': "layer_norm",
            'from': ["_conv_depthwise"]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    })

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_conv_output"


def make_conv_mod_011_se_block_and_pointwise_dim( # TODO
    net = None,
    in_l = None,
    prefix = None,

    # Layer specific args:
    kernel_size = None,
    conv_act = None,
    conv_post_dropout = None,
    use_se_block = False,
    pointwise_dim=None,

    # Shared network args:
    initialization = None,
    model_dim = None,

):

    if not pointwise_dim:
        pointwise_dim = 2 * model_dim

    from .network_additions import se_block
    net_add = {
        "_conv_laynorm" : {
            'class': "layer_norm",
            'from': [in_l]},
        "_conv_pointwise1" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_laynorm"], 
            'n_out': pointwise_dim,
            'forward_weights_init': initialization},
        "_conv_GLU" : {
            'class': "gating",
            'activation': "identity",
            'from': ["_conv_pointwise1"]},
        "_conv_depthwise": {
            'activation': None,
            'class': 'conv',
            'filter_size': (kernel_size,),
            'from': ['_conv_GLU'],
            'groups': model_dim,
            'n_out': model_dim,
            'padding': 'same',
            'with_bias': True}
    }

    next_layer = "_conv_depthwise"

    if use_se_block:
        net_add, next_layer = se_block(net_add, next_layer, prefix="_conv", model_dim=model_dim)

    net_add2 = {
        "_conv_batchnorm" : {
            'class': "layer_norm",
            'from': [next_layer]},
        "_conv_act" : {
            'class': "activation",
            'activation': conv_act,
            'from': ["_conv_batchnorm"]},
        "_conv_pointwise2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': False,
            'from': ["_conv_act"], 
            'n_out': model_dim,
            'forward_weights_init': initialization},
        "_conv_dropout" : {
            'class': "dropout",
            'dropout': conv_post_dropout,
            'from': ["_conv_pointwise2"]},
        "_conv_output" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_conv_dropout"], 
            'n_out': model_dim},
    }

    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )


    return net, f"{prefix}_conv_output"