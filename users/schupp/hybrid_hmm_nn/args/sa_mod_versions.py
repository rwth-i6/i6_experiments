from .conformer_returnn_dict_network_generator import prefix_all_keys

def make_self_att_mod_001(
    net = None,
    in_l = None,
    prefix = None,

    # Self attention args:
    num_heads = None,
    key_dim = None,
    value_dim = None,
    attention_left_only = None,
    sa_dropout = None,
    linear_mapping_bias = None,
    sa_post_dropout = None,

    # Shared args:
    initialization = None,
    model_dim = None,

):

    net_add = {
        "_self_att_laynorm": {
            'class': "layer_norm",
            'from': [in_l]},
        "_self_att_att": {
            'class': "self_attention", 
            'num_heads': num_heads,
            'total_key_dim': key_dim, 
            'n_out': value_dim,
            'from': ["_self_att_laynorm"],
            'attention_left_only': attention_left_only,
            'attention_dropout': sa_dropout,
            'forward_weights_init': initialization},
        "_self_att_lin" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': linear_mapping_bias,
            'from': ["_self_att_att"], 
            'n_out': model_dim,
            'forward_weights_init': initialization },
        "_self_att_drop" : {
            'class': "dropout", 
            'dropout': sa_post_dropout,
            'from': ["_self_att_att"]}, # TODO ERRROR! fix # We skipped the linear layer
        "_self_att_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_self_att_drop"],
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_self_att_out"


# + Positional Encoding
def make_self_att_mod_002(
    net = None,
    in_l = None,
    prefix = None,

    # Self attention args:
    num_heads = None,
    key_dim = None,
    value_dim = None,
    attention_left_only = None,
    sa_dropout = None,
    linear_mapping_bias = None,
    sa_post_dropout = None,
    fixed_pos = None,
    clipping = None,

    # Shared args:
    initialization = None,
    model_dim = None,


):

    net_add = {
        "_self_att_laynorm": {
            'class': "layer_norm",
            'from': [in_l]},
        '_rel_pos': { 
            'class': 'relative_positional_encoding',
            'clipping': clipping,
            'fixed': fixed_pos,
            'forward_weights_init': initialization,
            'from': ['_self_att_laynorm'],
            'n_out': key_dim // num_heads},
        "_self_att_att": {
            'class': "self_attention", 
            'num_heads': num_heads,
            'total_key_dim': key_dim, 
            'n_out': value_dim,
            'from': ["_self_att_laynorm"],
            'attention_left_only': attention_left_only,
            'attention_dropout': sa_dropout,
            'key_shift' : f"{prefix}_rel_pos",
            'forward_weights_init': initialization},
        "_self_att_lin" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': linear_mapping_bias,
            'from': ["_self_att_att"], 
            'n_out': model_dim,
            'forward_weights_init': initialization },
        "_self_att_drop" : {
            'class': "dropout", 
            'dropout': sa_post_dropout,
            'from': ["_self_att_att"]}, # ERR, this is fixed in the next itteration
        "_self_att_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_self_att_drop"],
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_self_att_out"


# + Positional Encoding
def make_self_att_mod_003_rel_pos(
    net = None,
    in_l = None,
    prefix = None,

    # Self attention args:
    num_heads = None,
    key_dim = None,
    value_dim = None,
    attention_left_only = None,
    sa_dropout = None,
    linear_mapping_bias = None,
    sa_post_dropout = None,
    fixed_pos = None,
    clipping = None,

    # Shared args:
    initialization = None,
    model_dim = None,


):

    net_add = {
        "_self_att_laynorm": {
            'class': "layer_norm",
            'from': [in_l]},
        '_rel_pos': { 
            'class': 'relative_positional_encoding',
            'clipping': clipping,
            'fixed': fixed_pos,
            'forward_weights_init': initialization,
            'from': ['_self_att_laynorm'],
            'n_out': key_dim // num_heads},
        "_self_att_att": {
            'class': "self_attention", 
            'num_heads': num_heads,
            'total_key_dim': key_dim, 
            'n_out': value_dim,
            'from': ["_self_att_laynorm"],
            'attention_left_only': attention_left_only,
            'attention_dropout': sa_dropout,
            'key_shift' : f"{prefix}_rel_pos",
            'forward_weights_init': initialization},
        "_self_att_lin" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': linear_mapping_bias,
            'from': ["_self_att_att"], 
            'n_out': model_dim,
            'forward_weights_init': initialization },
        "_self_att_drop" : {
            'class': "dropout", 
            'dropout': sa_post_dropout,
            'from': ["_self_att_lin"]}, # See fixed here
        "_self_att_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_self_att_drop"],
            'n_out': model_dim},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_self_att_out"





def make_self_att_mod_004_stoch_depth(
    net = None,
    in_l = None,
    prefix = None,

    # Self attention args:
    num_heads = None,
    key_dim = None,
    value_dim = None,
    attention_left_only = None,
    sa_dropout = None,
    linear_mapping_bias = None,
    sa_post_dropout = None,
    fixed_pos = None,
    clipping = None,

    survival_prob = None,

    # Shared args:
    initialization = None,
    model_dim = None,


):

    subnet = {
        "_self_att_laynorm": {
            'class': "layer_norm",
            'from': [in_l]},
        '_rel_pos': { 
            'class': 'relative_positional_encoding',
            'clipping': clipping,
            'fixed': fixed_pos,
            'forward_weights_init': initialization,
            'from': ['_self_att_laynorm'],
            'n_out': key_dim // num_heads},
        "_self_att_att": {
            'class': "self_attention", 
            'num_heads': num_heads,
            'total_key_dim': key_dim, 
            'n_out': value_dim,
            'from': ["_self_att_laynorm"],
            'attention_left_only': attention_left_only,
            'attention_dropout': sa_dropout,
            'key_shift' : f"{prefix}_rel_pos",
            'forward_weights_init': initialization},
        "_self_att_lin" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': linear_mapping_bias,
            'from': ["_self_att_att"], 
            'n_out': model_dim,
            'forward_weights_init': initialization },
        "_self_att_drop" : {
            'class': "dropout", 
            'dropout': sa_post_dropout,
            'from': ["_self_att_lin"]}, # See fixed here
        "_self_att_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_self_att_drop"],
            'n_out': model_dim},
    }

    from .network_additions import stochatic_depth_02_speedup

    prefixed_subnet = prefix_all_keys(prefix, subnet, in_l)

    net_sd, out = stochatic_depth_02_speedup(
        subnetwork=prefixed_subnet,
        survival_prob=survival_prob,
        subnet_last=f"{prefix}_self_att_out",
        in_l = in_l,

        prefix=f"{prefix}_att"
    )

    net.update(net_sd)

    return net, out



def make_self_att_mod_005_se_block(
    net = None,
    in_l = None,
    prefix = None,

    # Self attention args:
    num_heads = None,
    key_dim = None,
    value_dim = None,
    attention_left_only = None,
    sa_dropout = None,
    linear_mapping_bias = None,
    sa_post_dropout = None,
    fixed_pos = None,
    clipping = None,
    use_se_block = False,

    # Shared args:
    initialization = None,
    model_dim = None,


):

    from .network_additions import se_block
    net_add = {
        "_self_att_laynorm": {
            'class': "layer_norm",
            'from': [in_l]},
        '_rel_pos': { 
            'class': 'relative_positional_encoding',
            'clipping': clipping,
            'fixed': fixed_pos,
            'forward_weights_init': initialization,
            'from': ['_self_att_laynorm'],
            'n_out': key_dim // num_heads},
        "_self_att_att": {
            'class': "self_attention", 
            'num_heads': num_heads,
            'total_key_dim': key_dim, 
            'n_out': value_dim,
            'from': ["_self_att_laynorm"],
            'attention_left_only': attention_left_only,
            'attention_dropout': sa_dropout,
            'key_shift' : f"{prefix}_rel_pos",
            'forward_weights_init': initialization},
        "_self_att_lin" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': linear_mapping_bias,
            'from': ["_self_att_att"], 
            'n_out': model_dim,
            'forward_weights_init': initialization },
    }


    next_layer = "_self_att_lin"

    if use_se_block:
        net_add, next_layer = se_block(net_add, next_layer, prefix="_sa", model_dim=model_dim)

    net_add2 = {
        "_self_att_drop" : {
            'class': "dropout", 
            'dropout': sa_post_dropout,
            'from': [next_layer]}, # See fixed here
        "_self_att_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_self_att_drop"],
            'n_out': model_dim},
    }

    net_add.update(net_add2)

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_self_att_out"


# + goupnorm instead layer norm 
def make_self_att_mod_006_groupnorm(
    net = None,
    in_l = None,
    prefix = None,

    # Self attention args:
    num_heads = None,
    key_dim = None,
    value_dim = None,
    attention_left_only = None,
    sa_dropout = None,
    linear_mapping_bias = None,
    sa_post_dropout = None,
    fixed_pos = None,
    clipping = None,

    # Groupnorm args
    groups = None,
    epsilon = None,

    # Shared args:
    initialization = None,
    model_dim = None,


):

    from .network_additions import group_normalization

    net_add = {}
    net_add, _next = group_normalization(net_add, in_l, "_self_att", groups, epsilon, model_dim)

    net_add.update({
        '_rel_pos': { 
            'class': 'relative_positional_encoding',
            'clipping': clipping,
            'fixed': fixed_pos,
            'forward_weights_init': initialization,
            'from': [_next],
            'n_out': key_dim // num_heads},
        "_self_att_att": {
            'class': "self_attention", 
            'num_heads': num_heads,
            'total_key_dim': key_dim, 
            'n_out': value_dim,
            'from': [_next],
            'attention_left_only': attention_left_only,
            'attention_dropout': sa_dropout,
            'key_shift' : f"{prefix}_rel_pos",
            'forward_weights_init': initialization},
        "_self_att_lin" : {
            'class': "linear", 
            'activation': None, 
            'with_bias': linear_mapping_bias,
            'from': ["_self_att_att"], 
            'n_out': model_dim,
            'forward_weights_init': initialization },
        "_self_att_drop" : {
            'class': "dropout", 
            'dropout': sa_post_dropout,
            'from': ["_self_att_lin"]}, # See fixed here
        "_self_att_out" : {
            'class': "combine", 
            'kind': "add",
            'from': [in_l, "_self_att_drop"],
            'n_out': model_dim},
    })

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_self_att_out"
