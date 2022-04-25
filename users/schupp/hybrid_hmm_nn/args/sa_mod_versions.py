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
            'from': ["_self_att_att"]},
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
