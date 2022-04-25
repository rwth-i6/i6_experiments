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