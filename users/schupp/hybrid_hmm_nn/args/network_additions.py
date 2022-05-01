from .conformer_returnn_dict_network_generator import prefix_all_keys

def se_block( # TODO naming convention off here, should have '_00' cause inital implementation
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

def stochatic_depth_00( # TODO (WIP) this is unfinished!!
    subnetwork = None,
    survival_prob = None,
    subnet_last = None,

    in_l = None,

    prefix = None,

):

    random_bernulli = f"tf.compat.v1.distributions.Bernoulli(probs={survival_prob}).sample(sample_shape=())"
    switch = f"tf.equal({random_bernulli}, 0)"

    net_add = {
        f"{prefix}_train_flag" : {
            "class": "train_flag"},
        f"{prefix}_switch_train" : {
            "class" : "switch",
            "condition" : f"{prefix}_train_flag",
            "true_from" : f"{prefix}_stoch_depth_in_train",
            "false_from" : f"{prefix}_stoch_depth_in_eval"},
        f"{prefix}_stoch_depth_in_eval" : { # Then only multipy bu surival prob
            "class": "subnetwork",
            "from" : in_l,
            "subnetwork" : {
                in_l : { # We just copy this overunder the same name ( need it cause also used as input from the subnet )
                    "class" : "copy",
                    "from" : "data"},
                **subnetwork,
                "output" : {
                    "class": "eval",
                    "from" : [subnet_last, in_l], # TODO subnet last
                    "eval" : f"source(0) * {survival_prob} + source(1)"}}},
        f"{prefix}_stoch_depth_in_train": {
            "class": "cond", 
            "from": [],
            "condition": { # First condition only checks if we are in train using TrainFlagLayer
                "class": "eval", 
                "from": [], 
                "out_type": {
                    "batch_dim_axis": None, 
                    "shape": (), 
                    "dtype": "bool"},
                "eval": switch }, # In training generate random bernulli with 'surival_prob' if 0, then skip layer, if 1 the use layer ...
            "true_layer": { # TRUE add subnetwork output to redidual ( in_l )
                "class": "subnetwork", 
                "from": in_l, 
                "subnetwork": {
                    in_l : { # We just copy this overunder the same name
                        "class" : "copy",
                        "from" : "data"}, # TODO: we 
                    **subnetwork, # Most likely a full confore module
                    "output" : {
                        "class": "eval",
                        "from" : [subnet_last, in_l],
                        "eval" : "source(0) + source(1)"}}}, 
            "false_layer": { # FALSE: only add the residual i.e.: only 1 * input
                "class": "copy", 
                "from": in_l}},
    }


    return net_add, f"{prefix}_switch_train"
