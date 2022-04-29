from .conformer_returnn_dict_network_generator import prefix_all_keys

def se_block(
    net=None,
    in_l=None,
    prefix = None,

    # Shared args:
    model_dim = None,
):

    assert net, "no net"
    assert in_l, "no input layer"
    assert prefix, "needs prefix"

    net_add = {
        "_SE_reduce": {
            "class" : "reduce",
            "mode" : "mean",
            "from"  : in_l,
            "axes" : "T"},
        "_SE_linear1": {
            "class" : "linear",
            "from" : "_SE_reduce",
            "n_out" : 32},
        "_SE_act1" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear1"},
        "_SE_linear2" : {
            "class" : "linear",
            "from" :  "_SE_act1",
            "n_out" : model_dim},
        "_SE_act2" : {
            "class" : "activation",
            "activation" : "swish",
            "from" : "_SE_linear2" },
        "_SE_elm_mul" :  {
            "class" : "eval",
            "eval" : "source(0) * source(1)",
            "from" : ["_SE_act2", in_l]},
    }

    net.update(
        prefix_all_keys(prefix, net_add, in_l)
    )

    return net, f"{prefix}_SE_elm_mul"
