from .conformer_returnn_dict_network_generator import prefix_all_keys

def make_ff_mod_001(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,

    # Model shared args
    model_dim = None,
    initialization = None
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
            'forward_weights_init': initialization },
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_conv1"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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


def make_ff_mod_002_sd(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,
    survival_prob = None,

    # Model shared args
    model_dim = None,
    initialization = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    subnet = {
        "_laynorm" : {
            'class': "layer_norm",
            'from': in_l},
        "_conv1" : {
            'class': "linear", 
            'activation': ff_activation,
            'with_bias': True,
            'from': ["_laynorm"],
            'n_out': ff_dim, 
            'forward_weights_init': initialization },
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_conv1"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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

    from .network_additions import stochatic_depth_00

    prefixed_subnet = prefix_all_keys(prefix, subnet, in_l)

    net_sd, out = stochatic_depth_00(
        subnetwork=prefixed_subnet,
        survival_prob=survival_prob,
        subnet_last=f"{prefix}_out",
        in_l = in_l,

        prefix=f"{prefix}_ff"
    )

    net.update(net_sd)

    return net, out


def make_ff_mod_003_sd02(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,
    survival_prob = None,

    # Model shared args
    model_dim = None,
    initialization = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    subnet = {
        "_laynorm" : {
            'class': "layer_norm",
            'from': in_l}, # TODO: set *all name_scope = "/..." to share params
        "_conv1" : {
            'class': "linear", 
            'activation': ff_activation,
            'with_bias': True,
            'from': ["_laynorm"],
            'n_out': ff_dim, 
            'forward_weights_init': initialization },
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_conv1"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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

    from .network_additions import stochatic_depth_02_speedup

    prefixed_subnet = prefix_all_keys(prefix, subnet, in_l)

    net_sd, out = stochatic_depth_02_speedup(
        subnetwork=prefixed_subnet,
        survival_prob=survival_prob,
        subnet_last=f"{prefix}_out",
        in_l = in_l,

        prefix=f"{prefix}_ff"
    )

    net.update(net_sd)

    return net, out



def make_ff_mod_004_se_block(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,
    use_se_block = False,

    # Model shared args
    model_dim = None,
    initialization = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    from .network_additions import se_block

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
            'forward_weights_init': initialization }}


    next_layer = "_conv1"

    if use_se_block:
        net_add, next_layer = se_block(net_add, next_layer, prefix=f"_{prefix.split('_')[-1]}", model_dim=model_dim) # only the _ff1 from ff_ff1


    net_add2 = {
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_conv1"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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


    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_out"


def make_ff_mod_004_se_block_gating(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,
    use_se_block = False,
    gating_act = "identity",

    # Model shared args
    model_dim = None,
    initialization = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    from .network_additions import se_block

    net_add = {
        "_laynorm" : {
            'class': "layer_norm",
            'from': in_l},
        "_conv1" : {
            'class': "linear", 
            'activation': None,
            'with_bias': True,
            'from': ["_laynorm"],
            'n_out': ff_dim, 
            'forward_weights_init': initialization }}


    next_layer = "_conv1"

    if use_se_block:
        net_add, next_layer = se_block(net_add, next_layer, prefix=f"_{prefix.split('_')[-1]}", model_dim=model_dim) # only the _ff1 from ff_ff1



    net_add2 = {
        "_gate" : {
            'class': "gating",
            'activation': gating_act,
            'from': [f"_conv1"]},
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_gate"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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


    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_out"


def make_ff_mod_004_se_block_v2(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,
    use_se_block = False,

    # Model shared args
    model_dim = None,
    initialization = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    from .network_additions import se_block

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
            'forward_weights_init': initialization }}


    next_layer = "_conv1"

    if use_se_block:
        net_add, next_layer = se_block(net_add, next_layer, prefix=f"_{prefix.split('_')[-1]}", model_dim=ff_dim) # only the _ff1 from ff_ff1


    net_add2 = {
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': [next_layer], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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


    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_out"

def make_ff_mod_005_inital_groupnorm(
    net = None,
    in_l = None,
    prefix = None,

    # FF specific args
    ff_dim = None,
    ff_activation = None,
    ff_activation_dropout = None,
    ff_post_dropout = None,
    ff_half_ratio = None,

    # Groupnorm
    groups = None,
    epsilon = None,

    # Model shared args
    model_dim = None,
    initialization = None
):
    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"


    from .network_additions import group_normalization

    net_add = {}
    net_add, _next = group_normalization(net_add, in_l, "_ff", groups, epsilon, model_dim)

    net_add.update({
        "_conv1" : {
            'class': "linear", 
            'activation': ff_activation,
            'with_bias': True,
            'from': [_next],
            'n_out': ff_dim, 
            'forward_weights_init': initialization },
        "_conv2" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': True,
            'from': ["_conv1"], 
            'dropout': ff_activation_dropout,
            'n_out': model_dim, 
            'forward_weights_init': initialization},
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
    })

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_out"

