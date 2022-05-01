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